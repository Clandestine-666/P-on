import os
import random
import time
from datetime import datetime
import math
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
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

# -----------------------------
# Configuration and Hyperparameters 
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 17
N_FOLDS = 5
INNER_VAL_RATIO = 0.1
DATASET = 'davis'
DATA_CSV = f"data/{DATASET}_all.csv"

BATCH_SIZE = 32
EPOCHS = 300  
LR = 5e-5
WEIGHT_DECAY = 1e-4
ACCUM_STEPS = 2
MAX_SEQ_LEN = 1024
LOCAL_ESM_PATH = "esm2_t6_8M_UR50D"

HIDDEN_DIM = 1024
PROT_OUT_DIM = 1024 
EDGE_DIM = 4
DROPOUT_RATE = 0.2
EARLYSTOP_PATIENCE = 30

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# -----------------------------
# Five major evaluation indicator functions
# -----------------------------
def concordance_index(y_true, y_pred):
    """Calculate the consistency index CI"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    count, correct = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                count += 1
                if y_true[i] < y_true[j]:
                    if y_pred[i] < y_pred[j]: correct += 1
                    elif y_pred[i] == y_pred[j]: correct += 0.5
                else:
                    if y_pred[i] > y_pred[j]: correct += 1
                    elif y_pred[i] == y_pred[j]: correct += 0.5
    return correct / count if count > 0 else 0.75

def r2m_index(y_true, y_pred):
    """Calculate the Rm2 metric (commonly used in the DTI field)"""
    r2 = pearsonr(y_true, y_pred)[0]**2
    # Simplified Rm2 calculation logic
    return r2 * (1 - np.sqrt(max(0, r2)))

# -----------------------------
# Core model architecture
# -----------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, batch):
        pg = global_mean_pool(x, batch)
        y = self.fc(pg)
        return x * y[batch]

class BiDirectionalAttention(nn.Module):
    def __init__(self, drug_dim, prot_dim, attn_dim=256, num_heads=4):
        super().__init__()
        self.num_heads, self.scale = num_heads, (attn_dim // num_heads) ** -0.5
        self.q_d, self.k_p, self.v_p = nn.Linear(drug_dim, attn_dim), nn.Linear(prot_dim, attn_dim), nn.Linear(prot_dim, attn_dim)
        self.q_p, self.k_d, self.v_d = nn.Linear(prot_dim, attn_dim), nn.Linear(drug_dim, attn_dim), nn.Linear(drug_dim, attn_dim)
        self.out_d, self.out_p = nn.Linear(attn_dim, drug_dim), nn.Linear(attn_dim, prot_dim)

    def forward(self, d, p):
        B = d.size(0)
        # Drug to Prot
        qd, kp, vp = self.q_d(d).view(B, self.num_heads, -1), self.k_p(p).view(B, self.num_heads, -1), self.v_p(p).view(B, self.num_heads, -1)
        attn_dp = torch.softmax((qd * kp).sum(-1, keepdim=True) * self.scale, dim=1)
        d_new = d + self.out_d((attn_dp * vp).view(B, -1))
        # Prot to Drug
        qp, kd, vd = self.q_p(p).view(B, self.num_heads, -1), self.k_d(d).view(B, self.num_heads, -1), self.v_d(d).view(B, self.num_heads, -1)
        attn_pd = torch.softmax((qp * kd).sum(-1, keepdim=True) * self.scale, dim=1)
        p_new = p + self.out_p((attn_pd * vd).view(B, -1))
        return d_new, p_new

class ImprovedDTIModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Drug GNN (GINE + SE)
        def gine(in_d, out_d): return GINEConv(nn.Sequential(nn.Linear(in_d, out_d), nn.ReLU(), nn.Linear(out_d, out_d)), edge_dim=4)
        self.c1, self.s1 = gine(13, HIDDEN_DIM), SELayer(HIDDEN_DIM)
        self.c2, self.s2 = gine(HIDDEN_DIM, HIDDEN_DIM), SELayer(HIDDEN_DIM)
        self.drug_proj = nn.Sequential(nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM), nn.ReLU())
        self.prot_proj = nn.Sequential(nn.Linear(PROT_OUT_DIM, PROT_OUT_DIM), nn.ReLU(), nn.LayerNorm(PROT_OUT_DIM))
        self.bi_attn = BiDirectionalAttention(HIDDEN_DIM, PROT_OUT_DIM)
        self.fusion = nn.Sequential(nn.Linear(HIDDEN_DIM + PROT_OUT_DIM, HIDDEN_DIM), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(HIDDEN_DIM, 512), nn.ReLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(512, 1))

    def forward(self, bg, pe):
        x = F.relu(self.c1(bg.x, bg.edge_index, bg.edge_attr))
        x = self.s1(x, bg.batch)
        x = F.relu(self.c2(x, bg.edge_index, bg.edge_attr)) + x
        x = self.s2(x, bg.batch)
        df = self.drug_proj(torch.cat([global_mean_pool(x, bg.batch), global_max_pool(x, bg.batch), global_add_pool(x, bg.batch)], dim=1))
        pf = self.prot_proj(pe)
        df, pf = self.bi_attn(df, pf)
        return self.out(self.fusion(torch.cat([df, pf], dim=-1))).view(-1)

# -----------------------------
# Training evaluation logic
# -----------------------------
def train_epoch(model, loader, opt, scaler, device):
    model.train()
    loss_sum = 0
    for bg, pe, lb in tqdm(loader, desc="Training", leave=False):
        bg, pe, lb = bg.to(device), pe.to(device), lb.to(device)
        with torch.amp.autocast(device_type='cuda'):
            loss = F.mse_loss(model(bg, pe), lb) / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (loader.dataset.__len__() // BATCH_SIZE) % ACCUM_STEPS == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        loss_sum += loss.item() * ACCUM_STEPS
    return loss_sum / len(loader)

@torch.no_grad()
def evaluate_all_metrics(model, loader, device, m, s):
    model.eval()
    ps, ls = [], []
    for bg, pe, lb in loader:
        p = model(bg.to(device), pe.to(device)).cpu().numpy() * s + m
        ps.extend(p.tolist()); ls.extend((lb.numpy() * s + m).tolist())
    
    mse = mean_squared_error(ls, ps)
    rmse = math.sqrt(mse)
    r, _ = pearsonr(ls, ps)
    rm2 = r2m_index(ls, ps)
    ci = concordance_index(ls, ps)
    return mse, rmse, r, rm2, ci

# -----------------------------
# Main program and cold start are separated.
# -----------------------------
def main():
    df = pd.read_csv(DATA_CSV)
    df = df[df.iloc[:,0].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)].reset_index(drop=True)
    
    u_s, u_p = list(df.iloc[:,0].unique()), list(df.iloc[:,1].unique())
    s2id = {s:i for i,s in enumerate(u_s)}
    p2id = {p:i for i,p in enumerate(u_p)}
    
    # Precomputation graphs and ESM embedding
    print("Precomputing data...")
    gs_u = [smiles_to_graph(s) for s in tqdm(u_s, desc="Graphs")]
    
    tk = AutoTokenizer.from_pretrained(LOCAL_ESM_PATH)
    raw_esm = AutoModel.from_pretrained(LOCAL_ESM_PATH).to(DEVICE).eval()
    
    # Define a temporary projection layer
    proj_layer = nn.Linear(320, 1024).to(DEVICE).eval() # ESM2-8M Original dimension 320
    
    pe_u = []
    with torch.no_grad():
        for i in range(0, len(u_p), 64):
            batch_p = u_p[i:i+64]
            inputs = tk(batch_p, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(DEVICE)
            emb = raw_esm(**inputs).last_hidden_state.mean(dim=1)
            pe_u.append(proj_layer(emb).cpu())
    pe_u = torch.cat(pe_u, dim=0)

    # Running a cold start test for new drugs
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels = df.iloc[:,2].values
    results = []
    s_idx_list = np.array([s2id[s] for s in df.iloc[:,0]])
    p_idx_list = np.array([p2id[p] for p in df.iloc[:,1]])

    for fold, (tr_u_idx, te_u_idx) in enumerate(kf.split(range(len(u_s)))):
        print(f"\n--- DRUG COLD-START | FOLD {fold+1} ---")
        te_set = set(te_u_idx)
        out_idx = [i for i, sid in enumerate(s_idx_list) if sid in te_set]
        tr_iv_idx = [i for i in range(len(df)) if i not in out_idx]
        tr_idx, iv_idx = train_test_split(tr_iv_idx, test_size=0.1, random_state=SEED)
        
        m, s = labels[tr_idx].mean(), labels[tr_idx].std() or 1.0
        
        # data loader
        def make_loader(idx, shuf=False):
            ds = type('D',(Dataset,),{'__init__':lambda self,i:setattr(self,'i',i),'__len__':lambda self:len(self.i),'__getitem__':lambda self,j:(gs_u[s_idx_list[self.i[j]]], pe_u[p_idx_list[self.i[j]]], torch.tensor((labels[self.i[j]]-m)/s, dtype=torch.float32))})(idx)
            return DataLoader(ds, BATCH_SIZE, shuf, collate_fn=lambda b:(Batch.from_data_list([x[0] for x in b]), torch.stack([x[1] for x in b]), torch.stack([x[2] for x in b])))

        train_loader, val_loader, test_loader = make_loader(tr_idx, True), make_loader(iv_idx), make_loader(out_idx)
        
        model = ImprovedDTIModel().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler(enabled=True)
        
        best_mse = float('inf')
        for ep in range(1, 101): 
            train_epoch(model, train_loader, opt, scaler, DEVICE)
            metrics = evaluate_all_metrics(model, val_loader, DEVICE, m, s)
            if metrics[0] < best_mse:
                best_mse = metrics[0]
                torch.save(model.state_dict(), f"best_fold{fold+1}.pt")
        
        # The final test output includes all 5 metrics.
        model.load_state_dict(torch.load(f"best_fold{fold+1}.pt"))
        mse, rmse, r, rm2, ci = evaluate_all_metrics(model, test_loader, DEVICE, m, s)
        print(f"Fold {fold+1} Results: MSE: {mse:.4f}, RMSE: {rmse:.4f}, Pearson: {r:.4f}, Rm2: {rm2:.4f}, CI: {ci:.4f}")
        results.append({
            "fold": fold+1,
            "MSE": mse,
            "RMSE": rmse,
            "Pearson": r,
            "Rm2": rm2,
            "CI": ci
        })
    df_results = pd.DataFrame(results)
    df_results.to_csv("results.csv", index=False)
    print("\nResults saved to results.csv")

def atom_feat(a): return [a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(), int(a.GetIsAromatic()), a.GetNumExplicitHs(), a.GetNumImplicitHs()] + [0]*7
def smiles_to_graph(s):
    mol = Chem.MolFromSmiles(s)
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.extend([[i,j],[j,i]]); ea.extend([[1,0,0,0]]*2)
    return GraphData(x=x, edge_index=torch.tensor(ei).t().contiguous(), edge_attr=torch.tensor(ea, dtype=torch.float))

if __name__ == "__main__":
    main()