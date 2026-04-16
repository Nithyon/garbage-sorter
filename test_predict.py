"""
test_predict.py — Test the best_model_phase2.keras on sample images.
"""

import os
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

# ── Load config ──────────────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

input_shape = tuple(config["model"]["input_shape"])
dropout_rate = config["model"]["dropout_rate"]
dense_units = config["model"]["dense_units"]
image_size = config["data"]["image_size"]

# ── Build architecture ───────────────────────────────────────────────────────
print("Building model architecture...")
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

# ── Load weights from Keras 2.10 HDF5 ───────────────────────────────────────
print("Loading trained weights from best_model_phase2...")
weights_path = r"outputs\models\best_model_phase2.h5"
f = h5py.File(weights_path, "r")

# Load EfficientNetB0 backbone weights
eff_group = f["efficientnetb0"]
loaded, skipped = 0, 0
for layer in base_model.layers:
    if layer.name in eff_group:
        layer_group = eff_group[layer.name]
        weight_names = sorted(layer_group.keys())
        saved_weights = {w: np.array(layer_group[w]) for w in weight_names}
        if not saved_weights:
            continue

        # Match saved weights to expected weights by shape/size
        expected = layer.get_weights()
        if not expected:
            continue

        # Build ordered weight list matching expected shapes
        matched = []
        used = set()
        for exp_w in expected:
            best_key = None
            for key in saved_weights:
                if key in used:
                    continue
                sw = saved_weights[key]
                # Exact match
                if sw.shape == exp_w.shape:
                    best_key = key
                    break
                # Same number of elements -> can reshape
                if sw.size == exp_w.size:
                    best_key = key
                    break
            if best_key is not None:
                w = saved_weights[best_key]
                if w.shape != exp_w.shape:
                    w = w.reshape(exp_w.shape)
                matched.append(w)
                used.add(best_key)
            else:
                break

        if len(matched) == len(expected):
            try:
                layer.set_weights(matched)
                loaded += 1
            except Exception as e:
                skipped += 1
        else:
            skipped += 1

# Load classification head weights
model.get_layer("head_dense").set_weights([
    np.array(f["head_dense"]["head_dense"]["kernel:0"]),
    np.array(f["head_dense"]["head_dense"]["bias:0"]),
])
model.get_layer("output").set_weights([
    np.array(f["output"]["output"]["kernel:0"]),
    np.array(f["output"]["output"]["bias:0"]),
])
f.close()

print(f"  Backbone: {loaded} layers loaded, {skipped} skipped")
print(f"  Head: 2 layers loaded")
print("[OK] Model ready!\n")

# ── Predict ──────────────────────────────────────────────────────────────────
def predict(image_path):
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
        "wet_prob": round(prob * 100, 2),
        "dry_prob": round((1 - prob) * 100, 2),
    }

# ── Run on test samples ─────────────────────────────────────────────────────
test_dir = "test_samples"
print("=" * 65)
print(f"  {'Image':<30} {'Prediction':<12} {'Confidence':>10}")
print("=" * 65)

for fname in sorted(os.listdir(test_dir)):
    fpath = os.path.join(test_dir, fname)
    if os.path.isfile(fpath):
        result = predict(fpath)
        label = result["class"].upper()
        emoji = "[W]" if result["class"] == "wet" else "[D]"
        print(f"  {emoji} {result['image']:<28} {label:<12} {result['confidence']:>8.1f}%")
        print(f"     +-- Dry: {result['dry_prob']}%  |  Wet: {result['wet_prob']}%")

print("=" * 65)
print("\n[OK] Prediction complete!")
