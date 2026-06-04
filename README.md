# SEG-BiDTA: Drug–Target Affinity Prediction Model

This repository contains the implementation of a deep learning model for drug–target affinity (DTA) prediction.  
The model combines molecular graph representation learning, protein sequence embedding, Squeeze-and-Excitation attention, bi-directional attention, and residual feature fusion to predict drug–target binding affinity.

## 1\. Project Overview

Drug–target affinity prediction aims to estimate the binding strength between a drug molecule and a target protein.  
In this project, drug molecules are represented by molecular graphs constructed from SMILES strings, while protein sequences are encoded using a pretrained ESM2 model. The extracted drug and protein features are then fused through attention-based modules for final affinity prediction.

The main components of the model include:

* **GINE-based molecular graph encoder** for learning drug molecular structures;
* **Squeeze-and-Excitation (SE) module** for enhancing important molecular graph features;
* **ESM2 protein encoder** for extracting protein sequence representations;
* **Bi-Directional Attention module** for modeling drug–protein interactions;
* **Gated Residual Fusion module** for integrating drug and protein features;
* **Five-fold cross-validation** for model evaluation.

## 2\. Repository Structure

```text
.
├── data/
│   └── davis\_all.csv
├── esm2\_t6\_8M\_UR50D/
│   └── pretrained ESM2 model files
├── workW.py
├── workC.py
└── README.md
```

### File Description

|File|Description|
|-|-|
|`workW.py`|Main training script for the standard five-fold experiment. It uses GINE, SE, ESM2 embeddings, bi-directional attention, and gated residual fusion.|
|`workC.py`|Drug cold-start experiment script. It evaluates the model's generalization ability on unseen drugs.|
|`data/davis\_all.csv`|Input dataset file. The default dataset is the Davis dataset.|
|`esm2\_t6\_8M\_UR50D/`|Local pretrained ESM2 model directory.|

## 3\. Environment Requirements

The code is implemented in Python and mainly depends on PyTorch, PyTorch Geometric, RDKit, Transformers, and Scikit-learn.

Recommended environment:

```bash
python >= 3.8
torch
torch-geometric
rdkit
transformers
scikit-learn
scipy
numpy
pandas
tqdm
```

You can install most dependencies using:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install rdkit
pip install transformers scikit-learn scipy numpy pandas tqdm
```

> Note: The installation of `torch-geometric` may depend on your CUDA and PyTorch versions. Please install the version that matches your local environment.

## 4\. Dataset Format

By default, the scripts read data from:

```text
data/davis\_all.csv
```

The CSV file should contain at least three columns:

|Column|Meaning|
|-|-|
|Column 1|Drug SMILES string|
|Column 2|Protein sequence|
|Column 3|Binding affinity label|

Example:

```csv
SMILES,Protein,Affinity
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE,7.5
```

Before training, invalid SMILES strings are automatically filtered out.

## 5\. Pretrained Protein Model

The scripts use a local ESM2 model path:

```python
LOCAL\_ESM\_PATH = "esm2\_t6\_8M\_UR50D"
```

Therefore, you need to place the pretrained ESM2 model files in the `esm2\_t6\_8M\_UR50D/` directory.  
If your model is stored in another location, modify the following line in the scripts:

```python
LOCAL\_ESM\_PATH = "your\_esm2\_model\_path"
```

## 6\. How to Run

### 6.1 Standard Five-Fold Experiment

Run:

```bash
python workW.py
```

This script performs five-fold cross-validation and saves the results in a timestamped folder, for example:

```text
result\_WS\_davis\_20260603\_120000/
```

The output folder contains:

```text
best\_f1.pt
best\_f2.pt
...
fold\_summary.csv
```

### 6.2 Drug Cold-Start Experiment

Run:

```bash
python workC.py
```

This script performs a drug cold-start experiment, where drugs in the test set are not seen during training.  
The final evaluation results are saved as:

```text
results.csv
```

## 7\. Evaluation Metrics

The model is evaluated using the following metrics:

|Metric|Meaning|
|-|-|
|MSE|Mean Squared Error|
|RMSE|Root Mean Squared Error|
|Pearson|Pearson correlation coefficient|
|Rm2|Regression-based metric commonly used in DTA prediction|
|CI|Concordance Index|

Lower MSE and RMSE indicate better prediction accuracy, while higher Pearson, Rm2, and CI indicate stronger prediction consistency and ranking ability.

## 8\. Model Framework

The overall workflow is:

```text
Drug SMILES
   ↓
Molecular graph construction with RDKit
   ↓
GINE + SE drug encoder
   ↓
Drug feature representation

Protein sequence
   ↓
ESM2 protein encoder
   ↓
Protein feature representation

Drug feature + Protein feature
   ↓
Bi-Directional Attention
   ↓
Gated Residual Fusion
   ↓
MLP prediction layer
   ↓
Predicted binding affinity
```

## 9\. Reproducibility

The random seed is fixed in the scripts:

```python
SEED = 17
```

The following libraries are seeded to improve reproducibility:

```python
random
numpy
torch
torch.cuda
```

The scripts also enable deterministic CuDNN behavior:

```python
torch.backends.cudnn.deterministic = True
```

## 10\. Notes

1. Make sure the dataset path is correct before running the scripts.
2. Make sure the local ESM2 model directory exists.
3. GPU training is recommended because ESM2 embedding and graph neural network training can be time-consuming.
4. If CUDA is available, the scripts will automatically use GPU; otherwise, they will run on CPU.
5. The batch size, learning rate, number of epochs, and early stopping patience can be modified in the hyperparameter section of each script.

## 11\. Citation

If this code is used in academic work, please cite the related methods or models, such as GINE, ESM2, and drug–target affinity prediction studies.

## 12\. License

This project is for academic research and course-related experiments only.

