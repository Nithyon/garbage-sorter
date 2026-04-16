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

## Quick Start — Run the Trained Model

> **⚠️ IMPORTANT:** The model file `best_model_phase2.keras` was saved with **TensorFlow 2.10 / Keras 2** in HDF5 format.
> Newer TensorFlow versions (2.13+) expect `.keras` files to be ZIP archives, causing a **"File not found"** error.
> Use the script below — it handles this automatically.

### Step 1 — Install Dependencies (Python 3.10 or 3.12)

```bash
# Make sure you use Python 3.10–3.12 (NOT 3.13/3.14 — TensorFlow doesn't support them yet)
pip install tensorflow Pillow numpy pyyaml h5py
```

### Step 2 — Place the Model File

Copy `best_model_phase2.keras` into the `outputs/models/` folder:

```
GarbageClassification/
└── outputs/
    └── models/
        └── best_model_phase2.keras   ← place here
```

Or create the folder and copy:

```bash
# Windows (PowerShell)
New-Item -ItemType Directory -Path "outputs\models" -Force
Copy-Item "C:\path\to\best_model_phase2.keras" "outputs\models\"

# Linux / Mac
mkdir -p outputs/models
cp /path/to/best_model_phase2.keras outputs/models/
```

### Step 3 — Run Prediction

```bash
# Predict a single image
python run_model.py --image path/to/image.jpg

# Predict all images in a folder
python run_model.py --folder path/to/images/

# Just test that the model loads
python run_model.py --test
```

### `run_model.py` — Standalone Inference Script

This script handles the Keras 2→3 weight compatibility automatically.
It is already included in the repo root. If missing, create it with the contents below:

```python
"""
run_model.py — Standalone inference script for Wet/Dry Waste Classification.

Handles the Keras 2.10 (HDF5) → Keras 3 (TF 2.13+) weight compatibility
automatically. No need to downgrade TensorFlow.

Usage:
    python run_model.py --image photo.jpg
    python run_model.py --folder images/
    python run_model.py --test
"""

import os, sys, argparse
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import yaml

CLASS_NAMES = ["dry", "wet"]
MODEL_PATH = os.path.join("outputs", "models", "best_model_phase2.keras")
CONFIG_PATH = os.path.join("config", "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_and_load_model(config):
    """Build EfficientNetB0 architecture and load Keras 2.10 HDF5 weights."""
    input_shape = tuple(config["model"]["input_shape"])
    dropout_rate = config["model"]["dropout_rate"]
    dense_units = config["model"]["dense_units"]

    # Build architecture
    base_model = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=input_shape, pooling="avg",
    )
    base_model.trainable = True

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate * 0.67, name="head_dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs, outputs, name="WetDryClassifier")

    # Load weights from Keras 2.10 HDF5 file
    # The .keras file is actually HDF5 format (not ZIP), so we
    # rename to .h5 for h5py to read it properly.
    h5_path = MODEL_PATH.replace(".keras", ".h5")
    if not os.path.exists(h5_path):
        import shutil
        shutil.copy2(MODEL_PATH, h5_path)

    f = h5py.File(h5_path, "r")

    # --- Backbone weights ---
    eff_group = f["efficientnetb0"]
    loaded = 0
    for layer in base_model.layers:
        if layer.name in eff_group:
            layer_group = eff_group[layer.name]
            weight_names = sorted(layer_group.keys())
            saved_weights = {w: np.array(layer_group[w]) for w in weight_names}
            expected = layer.get_weights()
            if not expected:
                continue
            # Match by shape/size (Keras 2 Dense → Keras 3 Conv2D for SE layers)
            matched = []
            used = set()
            for exp_w in expected:
                for key in saved_weights:
                    if key in used:
                        continue
                    sw = saved_weights[key]
                    if sw.shape == exp_w.shape or sw.size == exp_w.size:
                        matched.append(sw.reshape(exp_w.shape) if sw.shape != exp_w.shape else sw)
                        used.add(key)
                        break
            if len(matched) == len(expected):
                try:
                    layer.set_weights(matched)
                    loaded += 1
                except Exception:
                    pass

    # --- Head weights ---
    model.get_layer("head_dense").set_weights([
        np.array(f["head_dense"]["head_dense"]["kernel:0"]),
        np.array(f["head_dense"]["head_dense"]["bias:0"]),
    ])
    model.get_layer("output").set_weights([
        np.array(f["output"]["output"]["kernel:0"]),
        np.array(f["output"]["output"]["bias:0"]),
    ])
    f.close()

    print(f"Model loaded: {loaded} backbone layers + 2 head layers")
    return model


def predict_image(model, image_path, image_size=224):
    """Classify a single image as Wet or Dry."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    pred_class = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
    confidence = prob if prob >= 0.5 else 1.0 - prob

    return {
        "image": os.path.basename(image_path),
        "class": pred_class,
        "confidence": round(confidence * 100, 2),
        "dry_prob": round((1 - prob) * 100, 2),
        "wet_prob": round(prob * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Wet/Dry Waste Classifier")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--test", action="store_true", help="Test that the model loads")
    args = parser.parse_args()

    if not any([args.image, args.folder, args.test]):
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Copy best_model_phase2.keras into outputs/models/")
        sys.exit(1)

    config = load_config()
    model = build_and_load_model(config)
    img_size = config["data"]["image_size"]

    if args.test:
        print("Model loaded and ready!")
        return

    # Collect images
    images = []
    if args.image:
        images = [args.image]
    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in exts
        )

    if not images:
        print("No images found.")
        sys.exit(1)

    # Run predictions
    print("=" * 60)
    print(f"  {'Image':<30} {'Class':<8} {'Confidence':>10}")
    print("=" * 60)
    for img_path in images:
        r = predict_image(model, img_path, img_size)
        print(f"  {r['image']:<30} {r['class'].upper():<8} {r['confidence']:>8.1f}%")
        print(f"     Dry: {r['dry_prob']}%  |  Wet: {r['wet_prob']}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### "File not found" / "not a valid .keras zip file"

**Cause:** The model was saved with TF 2.10 (Keras 2) which uses HDF5 format for `.keras` files.
TF 2.13+ (Keras 3) expects `.keras` files to be ZIP archives.

**Fix:** Use `run_model.py` (included in repo) — it rebuilds the model architecture and loads
the HDF5 weights with automatic Keras 2→3 shape conversion.

### "No module named 'tensorflow'"

**Fix:** Make sure you're using Python 3.10 or 3.12 (not 3.14):
```bash
# Check available Python versions (Windows)
py --list

# Install with the correct version
py -3.12 -m pip install tensorflow

# Run with the correct version
py -3.12 run_model.py --image photo.jpg
```

### "ModuleNotFoundError: No module named 'serial'" (for IoT/hardware)

```bash
pip install pyserial
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
