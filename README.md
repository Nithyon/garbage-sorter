# 🗑️ Wet/Dry Waste Classification

A deep-learning pipeline that classifies waste images into **Wet** (organic/biodegradable) and **Dry** (recyclable/non-biodegradable) categories using transfer learning with **EfficientNetB0** on TensorFlow.

Includes a **Developer Dashboard** for live inference testing, interactive dataset exploration, and model insight visualisation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dashboard](#dashboard)
- [Setup & Installation](#setup--installation)

---

## Overview

Proper waste segregation is critical for effective recycling and reducing landfill usage. This project automates the classification of waste into two categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **Wet** | Organic, biodegradable waste | Food scraps, fruit peels, rotten vegetables, biological matter |
| **Dry** | Non-biodegradable, recyclable waste | Plastic, metal, glass, paper, cardboard, clothes, batteries |

### Key Features

- **Two-phase transfer learning** — EfficientNetB0 backbone with custom classification head
- **Automated data pipeline** — Download, clean, deduplicate, and split datasets
- **7 Kaggle datasets** — 50,000+ images merged with automatic category mapping
- **Developer Dashboard** — Interactive web UI for testing, exploring, and reviewing
- **TF Lite export** — Float32, Float16, and Int8 quantisation options
- **Evaluation suite** — Confusion matrix, ROC-AUC curve, classification report

---

## Project Structure

```
GarbageClassification/
├── api/                         # FastAPI web server
│   ├── __init__.py
│   ├── main.py                  # API entry + prediction endpoints
│   └── dashboard.py             # Dashboard data endpoints
├── config/
│   └── config.yaml              # Hyperparameters & dataset mappings
├── data/
│   ├── raw/                     # Downloaded raw datasets (auto-created)
│   └── processed/               # Cleaned & split data (auto-created)
│       ├── train/{wet,dry}/
│       ├── val/{wet,dry}/
│       └── test/{wet,dry}/
├── src/                         # Core ML pipeline
│   ├── download_data.py         # Kaggle dataset download
│   ├── prepare_data.py          # Cleaning, mapping, dedup, splitting
│   ├── dataset.py               # tf.data pipeline & augmentation
│   ├── model.py                 # EfficientNetB0 architecture
│   ├── train.py                 # Two-phase training loop
│   ├── evaluate.py              # Metrics & visualisations
│   ├── evaluate_external.py     # Test on external raw datasets
│   ├── predict.py               # Single/batch inference
│   └── convert_tflite.py        # TF Lite conversion
├── webapp/                      # Dashboard frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── outputs/
│   ├── models/                  # Saved model checkpoints
│   ├── logs/                    # Training logs (JSON + TensorBoard)
│   └── plots/                   # Evaluation plots & reports
├── main.py                      # CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

This project merges **7 Kaggle datasets** into a large, diverse training set. All category mappings are defined in `config/config.yaml`.

| # | Dataset | Slug | Classes | Mapping Logic |
|---|---------|------|---------|---------------|
| 1 | Waste Classification | `techsash/waste-classification-data` | Organic, Recyclable | O→wet, R→dry |
| 2 | Garbage Classification | `mostafaabla/garbage-classification` | 12 household categories | biological→wet, rest→dry |
| 3 | Garbage Classification 2 | `asdasdasasdas/garbage-classification` | glass, paper, cardboard, plastic, metal, trash | All→dry |
| 4 | Garbage Classification V2 | `sumn2u/garbage-classification-v2` | 10 categories | biological→wet, rest→dry |
| 5 | Biodegradable Classification | `rayhanzamzamy/non-and-biodegradable-waste-dataset` | Biodegradable, Non-biodegradable | B→wet, N→dry |
| 6 | Waste Segregation 2 | `aashidutt3/waste-segregation-image-dataset` | organic, inorganic, recyclable, etc. | organic/biodegradable→wet, rest→dry |
| 7 | Waste Classification 3 | `phenomsg/waste-classification` | organic, inorganic, recyclable, hazardous | organic/biodegradable→wet, rest→dry |

### Category Mapping Summary

The fundamental rule is: **biodegradable/organic → wet**, **everything else → dry**.

| Original Class | → Mapped To | Rationale |
|---|---|---|
| organic / biological / biodegradable | **Wet** | Biodegradable organic matter |
| recyclable / inorganic / non-recyclable | **Dry** | General non-biodegradable |
| paper, cardboard | **Dry** | Recyclable fibre products |
| metal, plastic | **Dry** | Recyclable materials |
| glass (all variants) | **Dry** | Recyclable glass |
| clothes, shoes | **Dry** | Non-biodegradable textiles |
| batteries | **Dry** | Non-biodegradable (hazardous) |
| trash | **Dry** | General non-biodegradable |
| hazardous | **Dry** | Non-biodegradable (edge case — see note) |

> **Note:** `hazardous` waste is technically neither wet nor dry. Mapping it to "dry" is a practical choice for binary classification since hazardous items are non-biodegradable.

---

## Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 (ImageNet pretrained, ~5.3M params)
    ↓
GlobalAveragePooling
    ↓
Dropout (0.3)
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1, Sigmoid)  →  Wet (1) / Dry (0)
```

### Training Strategy

| Phase | Epochs | Learning Rate | What's Trained |
|-------|--------|--------------|----------------|
| **Phase 1** — Feature Extraction | 15 | 1e-3 | Classification head only |
| **Phase 2** — Fine-Tuning | 35 | 1e-5 | Top backbone layers + head |

**Key techniques:**
- EfficientNet-specific preprocessing (rescale to [-1, 1])
- Data augmentation: flip, rotation (±15°), zoom (±15%), contrast, brightness
- Class weight balancing for imbalanced wet/dry distribution
- ReduceLROnPlateau + EarlyStopping callbacks
- Best-model checkpointing by validation accuracy

---

## Results

After training, evaluation outputs are saved to `outputs/plots/`:

- `training_history.png` — Loss & accuracy curves
- `confusion_matrix.png` — Heatmap of predictions vs actual
- `roc_curve.png` — ROC-AUC curve
- `classification_report.txt` — Precision, recall, F1 per class
- `evaluation_results.json` — Machine-readable metrics

**Expected performance:** ≥ 95% test accuracy with the default configuration.

---

## Dashboard

The project includes a full-featured **Developer Dashboard** served directly from the FastAPI backend.

### Launching

```bash
python main.py --serve
# Opens at http://localhost:8000
```

### Tabs

| Tab | Description |
|-----|-------------|
| **Classifier Tester** | Upload images or paste URLs for live Wet/Dry prediction with confidence bars |
| **Dataset Explorer** | Browse raw datasets by folder, filter by Wet/Dry and original category, view image metadata |
| **Model Insights** | View accuracy, ROC-AUC, precision/recall/F1, confusion matrix, training history plots, and architecture config |

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU with CUDA** (recommended) or CPU
- **~5 GB disk space** for all datasets + models
- **Kaggle API key** for automated downloads

### Step 1 — Clone & Setup Environment

```bash
git clone <repo-url>
cd GarbageClassification

# Create virtual environment
python -m venv venv

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate — Linux / WSL / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Configure Kaggle API

1. Create a [Kaggle account](https://www.kaggle.com/)
2. Go to [Account Settings → API](https://www.kaggle.com/settings) → **Create New Token**
3. Save the downloaded `kaggle.json` to:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

### Step 3 — GPU Setup

**Option A — WSL2 (recommended for latest TensorFlow):**
```bash
# Install inside WSL2 Ubuntu with CUDA toolkit
pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Option B — Native Windows:**
```bash
# TF 2.10 is the last version with native Windows GPU support
pip install tensorflow==2.10.*
```

### Step 4 — Run the Pipeline

```bash
# Full pipeline (download → prepare → train → evaluate)
python main.py --download --prepare --train --evaluate

# Individual steps
python main.py --download            # Download all 7 datasets from Kaggle
python main.py --prepare             # Clean, map, deduplicate, split
python main.py --train               # Two-phase training
python main.py --evaluate            # Evaluate on test set

# Inference
python main.py --predict path/to/image.jpg
python main.py --predict path/to/folder/

# Test on an external raw dataset
python main.py --evaluate-external garbage_classification_v2

# Export to TF Lite
python main.py --convert-tflite --quantize float16

# Launch Dashboard
python main.py --serve
```

### Step 5 — Launch Dashboard

```bash
python main.py --serve
# Visit http://localhost:8000 in your browser
```

---

## Configuration

All hyperparameters are in `config/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.image_size` | 224 | Input image dimensions |
| `data.split_ratios` | 70/15/15 | Train/val/test split |
| `training.batch_size` | 16 | Batch size |
| `training.phase1_epochs` | 15 | Head-only training epochs |
| `training.phase2_epochs` | 35 | Fine-tuning epochs |
| `training.phase1_lr` | 1e-3 | Phase 1 learning rate |
| `training.phase2_lr` | 1e-5 | Phase 2 learning rate |
| `model.dropout_rate` | 0.3 | Classifier head dropout |

### Adding New Datasets

1. Add a new entry under `datasets:` in `config/config.yaml`
2. Define the `slug`, `description`, and `mapping` (category → wet/dry)
3. Run: `python main.py --download --prepare --train`

---

## TF Lite Deployment

```bash
# Standard (float32)
python main.py --convert-tflite --quantize float32

# Dynamic range (float16) — ~50% smaller
python main.py --convert-tflite --quantize float16

# Full integer (int8) — smallest, for microcontrollers
python main.py --convert-tflite --quantize int8
```

Output models are saved to `outputs/tflite/`.

---

## License

This project is for educational purposes. Datasets are subject to their respective Kaggle licenses.
