# Federated Learning for TB Chest X-Ray Classification
**EfficientNet-B0 + FedProx + Parallel Top-K Gradient Compression**

FAST-NUCES Islamabad | ANN + PDC Course Project  
Student IDs: 23i-0101 & 23i-0105

---

## Overview

This project implements a federated learning framework for tuberculosis detection from chest X-ray images. It improves over the baseline adaptive aggregation framework (Haripriya et al., 2025) by combining three components:

1. **EfficientNet-B0** — lightweight pretrained backbone replacing ResNet-50/VGG-16
2. **FedProx** — proximal regularization to prevent client drift under non-IID data
3. **Parallel Top-K Gradient Compression (PDC)** — 4 CPU threads compress client gradients simultaneously before aggregation

---

## Results

| Method | Backbone | Val Accuracy | Test Accuracy | F1 |
|---|---|---|---|---|
| Haripriya et al. (2025) | ResNet | 95.5% | — | — |
| Reproduced Baseline | ResNet | 91.36% | — | 90.40% |
| **Proposed** | EfficientNet-B0 | **98.73%** | **97.46%** | **97.38%** |

PDC average speedup: **1.17×** over sequential compression  
PDC bandwidth reduction: **90%** per round (Top-K κ=10%)

---

## Project Structure

```
├── Practice_TB_research.ipynb   # Main training notebook (Google Colab)
├── logs_tb_xray.csv             # Proposed method training logs (per round)
├── reproduce_tb.csv             # Reproduced baseline logs (per round)
├── comparison_plots.py          # Visualization code for all comparison graphs
└── README.md
```

---

## Dataset

**TB Chest Radiography Database**  
Download from Kaggle: `https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset`

Expected folder structure in Google Drive:
```
/content/drive/MyDrive/fl_data/
└── TB_Chest_Radiography_Database/
    ├── Normal/
    └── Tuberculosis/
```

The code automatically splits into 70% train / 15% val / 15% test using stratified sampling.

---

## Setup

### Google Colab (Recommended)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install timm --break-system-packages -q
```

### Required Libraries

```
torch
torchvision
timm
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Configuration

All hyperparameters are controlled from the `Config` dataclass at the top of the notebook. Change `dataset_name` to switch datasets:

```python
@dataclass
class Config:
    dataset_name    : str   = "TB_Chest_Radiography_Database"
    data_root       : str   = "/content/drive/MyDrive/fl_data"
    save_dir        : str   = "/content/drive/MyDrive/fl_checkpoints"
    backbone        : str   = "efficientnet_b0"
    img_size        : int   = 224
    num_clients     : int   = 10
    num_rounds      : int   = 40
    local_epochs    : int   = 3
    batch_size      : int   = 32
    fedprox_mu      : float = 0.01     # proximal coefficient
    div_threshold   : float = 0.002    # FedAvg/FedSGD switching threshold
    threshold_ema   : float = 0.2      # EMA smoothing for adaptive τ
    compression     : float = 0.10     # Top-K ratio (keep top 10%)
    lr              : float = 3e-4
    weight_decay    : float = 1e-4
    dirichlet_alpha : float = 0.5      # non-IID degree (lower = more skewed)
    train_ratio     : float = 0.70
    val_ratio       : float = 0.15
    seed            : int   = 42
    mu              : float = 0.01
```

---

## How It Works

### 1. Non-IID Data Partitioning

Client data is split using Dirichlet(α=0.5) — a low α means highly skewed distributions across clients, simulating realistic hospital scenarios where different institutions see different patient populations.

```python
# Each client gets a different proportion of each class
partition = dirichlet_partition(train_dataset, cfg.num_clients,
                                alpha=cfg.dirichlet_alpha, seed=cfg.seed)
```

### 2. Local Training with FedProx

Each client minimizes:

```
Loss = CrossEntropy(predictions, labels) + (μ/2) * ||w - w_global||²
```

The second term (proximal term) prevents the local model from drifting too far from the global model. Without this, non-IID data causes the global model to diverge — as seen in the reproduced baseline which collapsed to 16% accuracy at round 20.

```python
# Proximal term computed per layer
prox_loss = sum(
    torch.norm(p - gp.to(DEVICE)) ** 2
    for p, gp in zip(model.parameters(), global_params_ref)
    if p.requires_grad
)
loss = ce_loss + (cfg.fedprox_mu / 2.0) * prox_loss
```

### 3. Parallel Top-K Gradient Compression (PDC)

After all clients train locally, their gradient deltas are compressed **in parallel** using 4 CPU threads before being sent to the server. This is the PDC contribution.

```python
def parallel_compress(all_client_params, global_params_cpu, ratio, current_round):
    args = [(k, cp, global_params_cpu, ratio)
            for k, cp in enumerate(all_client_params)]

    # Fork: dispatch all K compression jobs to T threads simultaneously
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="PDC-Worker") as executor:
        results = list(executor.map(compress_one_client, args))
    # Join: all threads synchronize here before aggregation
    return results
```

Each thread runs `compress_one_client` which:
- Computes gradient delta: `Δ = w_client - w_global`
- Finds top 10% values by magnitude
- Zeros out the remaining 90%
- Returns the sparse compressed update

### 4. Adaptive Aggregation

The server computes gradient divergence δₜ across all clients. If δₜ > τₜ it uses FedSGD (more stable), otherwise FedAvg (more efficient). The threshold τₜ updates automatically each round via EMA:

```python
# τ self-tunes to prevailing divergence level
tau = cfg.threshold_ema * divergence + (1 - cfg.threshold_ema) * tau
```

In our experiments FedAvg was selected in 38/40 rounds — FedSGD only triggered at rounds 36 and 38.

---

## Training Loop

The full per-round procedure:

```
For each round t:
  1. Broadcast global weights to all 10 clients
  2. Each client trains locally for 3 epochs (FedProx objective)
  3. PDC: 4 threads compress all 10 client deltas in parallel (Top-K 10%)
  4. Compute divergence δₜ across compressed updates
  5. Adaptive aggregation: FedAvg if δₜ ≤ τₜ, else FedSGD
  6. Update adaptive threshold τₜ via EMA
  7. Evaluate on test set, save best model to Drive
```

Round 1 takes ~14 minutes (model download + first epoch). Subsequent rounds take ~3-4 minutes each. Full 40-round run: ~3 hours on T4 GPU.

---

## Output Files

After training the following are saved to `save_dir`:

| File | Description |
|---|---|
| `best_TB_Chest_Radiography_Database.pt` | Best model checkpoint |
| `logs_tb_xray.csv` | Per-round: loss, val_acc, val_f1, divergence, method, pdc stats |
| `fig1_accuracy_*.png` | Accuracy + F1 convergence curve |
| `fig2_loss_*.png` | Training loss curve |
| `fig3_aggregation_*.png` | FedAvg/FedSGD per round + divergence plot |
| `fig4_confusion_*.png` | Confusion matrix (counts + normalized) |
| `fig5_distribution_*.png` | Non-IID client data distribution |
| `fig7_pdc_threads_*.png` | PDC thread timing + speedup analysis |

---

## Generating Comparison Plots

After you have both `logs_tb_xray.csv` (proposed) and `reproduce_tb.csv` (reproduced baseline), run the comparison plotting script locally:

```bash
pip install pandas matplotlib seaborn scikit-learn
python comparison_plots.py
```

Make sure both CSV files are in the same directory as the script. Output figures are saved to `./fl_results/`.

---

## PDC Thread Output

In round 1, the code prints full thread details to confirm parallelism is working:

```
[PDC] Spawning 4 threads for 10 clients | Top-K=10%
  [PDC] Thread=PDC-Worker_0 | ID=139758 | Client=0 | Kept=10.0% | Time=0.384s
  [PDC] Thread=PDC-Worker_1 | ID=139759 | Client=1 | Kept=10.0% | Time=0.337s
  [PDC] Thread=PDC-Worker_2 | ID=139760 | Client=2 | Kept=10.0% | Time=0.362s
  [PDC] Thread=PDC-Worker_3 | ID=139761 | Client=3 | Kept=10.0% | Time=0.388s
  ...
[PDC] All threads done in 0.91s
```

From round 2 onward only the summary line prints:
```
[PDC] 4 threads | 10 clients | Top-K=10% | Done in 1.17s
```

---

## Known Issues

- **First round slow (~14 min)**: EfficientNet-B0 downloads from HuggingFace on first run. Subsequent rounds are fast.
- **num_workers warning**: Set `NUM_WORKERS = 0` in the code to avoid Colab multiprocessing warnings.
- **Colab disconnect**: Free tier disconnects after ~90 min idle. Model saves to Drive after every improvement so progress is not lost.
- **Thread speedup variance**: PDC speedup ranges 0.76×–1.50× due to Python GIL behavior on CPU-bound tasks. The primary benefit is 90% bandwidth reduction, not raw speed.

---

## Citation

If you use this code, please cite the baseline paper:

```
R. Haripriya, N. Khare, M. Pandey, and S. Biswas,
"A privacy-enhanced framework for collaborative Big Data analysis
in healthcare using adaptive federated learning aggregation,"
Journal of Big Data, vol. 12, article no. 113, May 2025.
DOI: 10.1186/s40537-025-01169-8
```
