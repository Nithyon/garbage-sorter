"""
run_model.py - Standalone inference script for Wet/Dry Waste Classification.

Handles the Keras 2.10 (HDF5) -> Keras 3 (TF 2.13+) weight compatibility
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

    # --- Load weights from Keras 2.10 HDF5 file ---
    # The .keras file is actually HDF5 format (not ZIP), so we copy it
    # with a .h5 extension for h5py to read it properly.
    print("Loading trained weights...")
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
            # Match saved weights to expected weights by shape/size.
            # Keras 2 used Dense layers for SE blocks (2D weights),
            # Keras 3 uses Conv2D (4D weights) — we reshape automatically.
            matched = []
            used = set()
            for exp_w in expected:
                for key in saved_weights:
                    if key in used:
                        continue
                    sw = saved_weights[key]
                    if sw.shape == exp_w.shape or sw.size == exp_w.size:
                        matched.append(
                            sw.reshape(exp_w.shape) if sw.shape != exp_w.shape else sw
                        )
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

    print(f"  -> {loaded} backbone layers + 2 head layers loaded successfully")
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
    parser = argparse.ArgumentParser(
        description="Wet/Dry Waste Classifier - Standalone Inference",
        epilog="Examples:\n"
               "  python run_model.py --image photo.jpg\n"
               "  python run_model.py --folder test_images/\n"
               "  python run_model.py --test\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--test", action="store_true", help="Test that the model loads correctly")
    args = parser.parse_args()

    if not any([args.image, args.folder, args.test]):
        parser.print_help()
        sys.exit(0)

    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print()
        print("Fix: Copy best_model_phase2.keras into outputs/models/")
        print("  mkdir outputs\\models          (Windows)")
        print("  mkdir -p outputs/models        (Linux/Mac)")
        print("  Then copy the .keras file there.")
        sys.exit(1)

    # Check config exists
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file not found at '{CONFIG_PATH}'")
        sys.exit(1)

    config = load_config()
    model = build_and_load_model(config)
    img_size = config["data"]["image_size"]

    if args.test:
        print("\nModel loaded and ready! You can now use --image or --folder.")
        return

    # Collect images
    images = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        images = [args.image]
    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"ERROR: Folder not found: {args.folder}")
            sys.exit(1)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in exts
        )

    if not images:
        print("No valid images found.")
        sys.exit(1)

    # Run predictions
    print()
    print("=" * 60)
    print(f"  {'Image':<30} {'Class':<8} {'Confidence':>10}")
    print("=" * 60)
    for img_path in images:
        r = predict_image(model, img_path, img_size)
        tag = "[W]" if r["class"] == "wet" else "[D]"
        print(f"  {tag} {r['image']:<28} {r['class'].upper():<8} {r['confidence']:>8.1f}%")
        print(f"      Dry: {r['dry_prob']}%  |  Wet: {r['wet_prob']}%")
    print("=" * 60)

    # Summary
    wet = sum(1 for img in images for r in [predict_image(model, img, img_size)] if r["class"] == "wet")
    dry = len(images) - wet
    print(f"\n  Summary: {dry} dry, {wet} wet (total: {len(images)})")


if __name__ == "__main__":
    main()
