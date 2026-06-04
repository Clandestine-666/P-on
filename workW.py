import os
import random
import time
from datetime import datetime
import math
import json
import shutil

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GraphData, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

# -----------------------------
# Hyperparameters
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 17
N_FOLDS = 5
INNER_VAL_RATIO = 0.1
TASK_START = 'WS'
DATASET = 'davis'
DATA_CSV = f"data/{DATASET}_all.csv"

BATCH_SIZE = 32
EPOCHS = 500
LR = 5e-5
WEIGHT_DECAY = 1e-4
ACCUM_STEPS = 2
MAX_SEQ_LEN = 1024
LOCAL_ESM_PATH = "esm2_t6_8M_UR50D"

GRAD_CLIP_NORM = 1.0
PRE_COMPUTE_BATCH_SIZE = 64

HIDDEN_DIM = 1024
PROT_OUT_DIM = 1024
EDGE_DIM = 4
DROPOUT_RATE = 0.2
EARLYSTOP_PATIENCE = 30

NUM_WORKERS = max(1, (os.cpu_count() or 2) // 2)
PIN_MEMORY = True if DEVICE.startswith("cuda") else False
SAVE_MODELS = True

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# -----------------------------
# 1:Squeeze-and-Excitation
# -----------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, batch):
        # x: [N, C], batch: [N]
        pg = global_mean_pool(x, batch)  # [BatchSize, C]
        y = self.fc(pg)                  # [BatchSize, C]
        return x * y[batch]              # Broadcast to every atomic node

# -----------------------------
# 2:Bi-Directional Attention
# -----------------------------
class BiDirectionalAttention(nn.Module):
    def __init__(self, drug_dim, prot_dim, attn_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (attn_dim // num_heads) ** -0.5
        
        # Drug -> Protein
        self.q_drug = nn.Linear(drug_dim, attn_dim)
        self.k_prot = nn.Linear(prot_dim, attn_dim)
        self.v_prot = nn.Linear(prot_dim, attn_dim)
        
        # Protein -> Drug
        self.q_prot = nn.Linear(prot_dim, attn_dim)
        self.k_drug = nn.Linear(drug_dim, attn_dim)
        self.v_drug = nn.Linear(drug_dim, attn_dim)
        
        self.proj_drug = nn.Linear(attn_dim, drug_dim)
        self.proj_prot = nn.Linear(attn_dim, prot_dim)

    def forward(self, d_feat, p_feat):
        B = d_feat.size(0)
        # D -> P Attention
        qd = self.q_drug(d_feat).view(B, self.num_heads, -1)
        kp = self.k_prot(p_feat).view(B, self.num_heads, -1)
        vp = self.v_prot(p_feat).view(B, self.num_heads, -1)
        attn_dp = torch.softmax((qd * kp).sum(-1, keepdim=True) * self.scale, dim=1)
        out_d = self.proj_drug((attn_dp * vp).view(B, -1))
        
        # P -> D Attention
        qp = self.q_prot(p_feat).view(B, self.num_heads, -1)
        kd = self.k_drug(d_feat).view(B, self.num_heads, -1)
        vd = self.v_drug(d_feat).view(B, self.num_heads, -1)
        attn_pd = torch.softmax((qp * kd).sum(-1, keepdim=True) * self.scale, dim=1)
        out_p = self.proj_prot((attn_pd * vd).view(B, -1))
        
        return d_feat + out_d, p_feat + out_p

# -----------------------------
# 3: Gated Residual Fusion
# -----------------------------
class GatedResidualFusion(nn.Module):
    def __init__(self, d_dim, p_dim, out_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_dim + p_dim, out_dim), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(d_dim + p_dim, out_dim), nn.Tanh())
        self.res_proj = nn.Linear(d_dim, out_dim)

    def forward(self, d, p):
        combined = torch.cat([d, p], dim=-1)
        g = self.gate(combined)
        v = self.transform(combined)
        return g * v + (1 - g) * self.res_proj(d)

# -----------------------------
# Basic Atom Chemical Transformation Logic 
# -----------------------------
HYBRID_TYPES = [rdchem.HybridizationType.S, rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
                rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2,
                rdchem.HybridizationType.UNSPECIFIED]
HYB_TO_IDX = {h: i for i, h in enumerate(HYBRID_TYPES)}
NODE_FEAT_DIM = 6 + len(HYBRID_TYPES)

def atom_features(atom):
    hyb_onehot = [0] * len(HYBRID_TYPES)
    idx = HYB_TO_IDX.get(atom.GetHybridization(), HYB_TO_IDX[rdchem.HybridizationType.UNSPECIFIED])
    hyb_onehot[idx] = 1
    return [atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic()),
            atom.GetNumExplicitHs(), atom.GetNumImplicitHs()] + hyb_onehot

def bond_features(bond):
    bt = bond.GetBondType()
    return [int(bt == Chem.rdchem.BondType.SINGLE), int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE), int(bt == Chem.rdchem.BondType.AROMATIC)]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return GraphData(x=torch.zeros((1, NODE_FEAT_DIM)), edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, EDGE_DIM)))
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.extend([[i, j], [j, i]]); ea.extend([bond_features(b), bond_features(b)])
    return GraphData(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(ea, dtype=torch.float))

# -----------------------------
# Model structure definition
# -----------------------------
class DrugGNN(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, edge_feat_dim=EDGE_DIM):
        super().__init__()
        def gine_block(in_d, out_d):
            return GINEConv(nn.Sequential(nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d), nn.ReLU(), nn.Linear(out_d, out_d), nn.ReLU()), edge_dim=edge_feat_dim)
        
        self.conv1 = gine_block(node_feat_dim, hidden_dim)
        self.se1 = SELayer(hidden_dim)
        self.conv2 = gine_block(hidden_dim, hidden_dim)
        self.se2 = SELayer(hidden_dim)
        self.conv3 = gine_block(hidden_dim, hidden_dim)
        self.se3 = SELayer(hidden_dim)
        self.res_p = nn.Linear(node_feat_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = F.relu(self.conv1(x, edge_index, edge_attr)) + self.res_p(x)
        x1 = self.se1(x1, batch)
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr)) + x1
        x2 = self.se2(x2, batch)
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr)) + x2
        x3 = self.se3(x3, batch)
        return torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch), global_add_pool(x3, batch)], dim=1)

class ProteinEncoder(nn.Module):
    def __init__(self, model_path=LOCAL_ESM_PATH, out_dim=PROT_OUT_DIM):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        for p in self.model.parameters(): p.requires_grad = False
        try:
            for p in self.model.encoder.layer[-6:].parameters(): p.requires_grad = True
        except: pass
        self.fc = nn.Linear(self.model.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(out.last_hidden_state.mean(dim=1))

class ImprovedDTIModelPrecomputed(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_enc = DrugGNN()
        self.drug_proj = nn.Sequential(nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM), nn.ReLU())
        self.prot_proj = nn.Sequential(nn.Linear(PROT_OUT_DIM, PROT_OUT_DIM), nn.ReLU(), nn.LayerNorm(PROT_OUT_DIM))
        
        #Core innovation module
        self.bi_attn = BiDirectionalAttention(HIDDEN_DIM, PROT_OUT_DIM)
        self.fusion = GatedResidualFusion(HIDDEN_DIM, PROT_OUT_DIM, HIDDEN_DIM)
        
        self.output = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(HIDDEN_DIM // 2, 1))

    def forward(self, batch_graph, prot_embeddings):
        d_f = self.drug_proj(self.drug_enc(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch))
        p_f = self.prot_proj(prot_embeddings)
        
        d_f, p_f = self.bi_attn(d_f, p_f)
        fused = self.fusion(d_f, p_f)
        return self.output(fused).view(-1)

# -----------------------------
# Training and evaluation functions
# -----------------------------
class DTIDatasetPrecomputed(Dataset):
    def __init__(self, gs, pe, ln, lr=None): self.gs, self.pe, self.ln, self.lr = gs, pe, ln, lr
    def __len__(self): return len(self.gs)
    def __getitem__(self, i): return self.gs[i], self.pe[i], torch.tensor(self.ln[i], dtype=torch.float32)

def collate_fn_precomputed(b):
    gs, pes, ls = zip(*b)
    return Batch.from_data_list(gs), torch.stack(pes), torch.stack(ls)

def train_epoch(model, loader, opt, sched, scaler, device):
    model.train()
    loss_sum = 0
    for i, (bg, pe, lb) in enumerate(tqdm(loader, desc="Train", leave=False)):
        bg, pe, lb = bg.to(device), pe.to(device), lb.to(device)
        with torch.amp.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu'):
            p = model(bg, pe)
            loss = F.mse_loss(p, lb) / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (i+1) % ACCUM_STEPS == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            if sched: sched.step()
        loss_sum += loss.item() * ACCUM_STEPS
    return loss_sum / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, m=0, s=1, return_preds=False):
    model.eval()
    ps, ls = [], []
    #The evaluation phase must either proceed with autocasting or ensure dtype consistency.
    with torch.amp.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu'):
        for bg, pe, lb in loader:
            p = model(bg.to(device), pe.to(device)).cpu().numpy() * s + m
            ps.extend(p.tolist()); ls.extend((lb.numpy() * s + m).tolist())
    mse = mean_squared_error(ls, ps)
    r, _ = pearsonr(ls, ps)
    if return_preds: return mse, math.sqrt(mse), r, np.array(ps), np.array(ls)
    return mse, math.sqrt(mse), r

# -----------------------------
# Main loop and save logic
# -----------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"result_{TASK_START}_{DATASET}_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    df = df[df.iloc[:, 0].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)].reset_index(drop=True)
    
    print("[INFO] Precomputing...")
    gs_all = [smiles_to_graph(s) for s in tqdm(df.iloc[:, 0], desc="Graphs")]
    tk = AutoTokenizer.from_pretrained(LOCAL_ESM_PATH)
    pm = ProteinEncoder().to(DEVICE)
    
    # Pre-computational protein embedding
    pe_all = []
    pm.eval()
    for i in range(0, len(df), PRE_COMPUTE_BATCH_SIZE):
        batch = df.iloc[i:i+PRE_COMPUTE_BATCH_SIZE, 1].tolist()
        enc = tk(batch, padding='max_length', truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): pe_all.append(pm(enc['input_ids'], enc['attention_mask']).cpu())
    pe_all = torch.cat(pe_all, dim=0)
    labels = df.iloc[:, 2].values

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_res = []

    for fold, (tv_idx, out_idx) in enumerate(kf.split(gs_all)):
        print(f"\n--- FOLD {fold+1} ---")
        tr_idx, iv_idx = train_test_split(tv_idx, test_size=INNER_VAL_RATIO, random_state=SEED)
        m, s = labels[tr_idx].mean(), labels[tr_idx].std() or 1.0

        def get_loader(idx, shuffle=False):
            ds = DTIDatasetPrecomputed([gs_all[i] for i in idx], pe_all[idx], (labels[idx]-m)/s)
            return DataLoader(ds, BATCH_SIZE, shuffle, collate_fn=collate_fn_precomputed, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        tr_loader, iv_loader, out_loader = get_loader(tr_idx, True), get_loader(iv_idx), get_loader(out_idx)
        
        model = ImprovedDTIModelPrecomputed().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # --- Optimized scheduler---
        # If the MSE does not improve for 10 consecutive epochs, the learning rate is halved.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=10, verbose=True
        )
        # ---------------------
        
        scaler = torch.amp.GradScaler(enabled=DEVICE=="cuda")
        best_iv_mse, patience = float('inf'), 0
        best_path = os.path.join(res_dir, f"best_f{fold+1}.pt")

        for ep in range(1, EPOCHS + 1):
            loss = train_epoch(model, tr_loader, opt, None, scaler, DEVICE)
            iv_mse, iv_rmse, iv_r, iv_rm2, iv_ci = evaluate(model, iv_loader, DEVICE, m, s)
            
            # Use the validation set MSE to drive the scheduler
            scheduler.step(iv_mse) 
            
            if iv_mse < best_iv_mse:
                best_iv_mse, patience = iv_mse, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            
            if ep % 20 == 0 or patience == 0:
                print(f"Ep {ep} | Loss: {loss:.4f} | IV_MSE: {iv_mse:.4f} | LR: {opt.param_groups[0]['lr']:.2e}")
            
            if patience >= EARLYSTOP_PATIENCE: break

        # Final evaluation and saving of results
        model.load_state_dict(torch.load(best_path))
        o_mse, o_rmse, o_r,o_rm2,o_ci, o_p, o_l = evaluate(model, out_loader, DEVICE, m, s, True)

        # Save the new metric to the results list.
        fold_res.append({
            "fold": fold+1, 
            "mse": o_mse, 
            "rmse": o_rmse, 
            "r": o_r, 
            "rm2": o_rm2, 
            "ci": o_ci
        })
        
        pd.DataFrame(fold_res).to_csv(os.path.join(res_dir, "fold_summary.csv"), index=False)
        print(f"Fold {fold+1} Final -> MSE: {o_mse:.4f}, RM2: {o_rm2:.4f}, CI: {o_ci:.4f}")

    print(f"\n[DONE] Results in: {res_dir}")

if __name__ == "__main__":
    main()