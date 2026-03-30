"""
convert_tflite.py — Convert the trained Keras model to TensorFlow Lite.

Supports:
  - Standard float32 conversion
  - Dynamic-range quantisation (float16)
  - Full integer quantisation (int8) for edge / microcontroller deployment
"""

import os
import logging

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras

from src.dataset import get_dataset, CLASS_NAMES

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_representative_dataset(config: dict):
    """
    Generator that yields representative samples for int8 quantisation.
    Uses a subset of the training data.
    """
    train_dir = os.path.join(config["data"]["processed_dir"], "train")
    img_size = config["data"]["image_size"]

    ds = get_dataset(
        train_dir,
        image_size=img_size,
        batch_size=1,
        shuffle=True,
        augment=False,
    )

    for images, _ in ds.take(200):
        yield [images]


def convert_to_tflite(
    config: dict,
    model_path: str | None = None,
    quantize: str = "float32",
) -> str:
    """
    Convert a saved Keras model to TFLite format.

    Args:
        config: Project config dict.
        model_path: Path to saved .keras model. Auto-detected if None.
        quantize: One of 'float32', 'float16', 'int8'.

    Returns:
        Path to the saved .tflite file.
    """
    tflite_dir = config["output"]["tflite_dir"]
    os.makedirs(tflite_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────
    if model_path is None:
        candidates = [
            os.path.join(config["output"]["models_dir"], "best_model_phase2.keras"),
            os.path.join(config["output"]["models_dir"], "final_model.keras"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Train the model first.")

    logger.info(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)

    # ── Convert ──────────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "float16":
        logger.info("Applying float16 quantisation ...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        suffix = "_float16"

    elif quantize == "int8":
        logger.info("Applying int8 full-integer quantisation ...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _get_representative_dataset(config)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        suffix = "_int8"

    else:
        logger.info("Standard float32 conversion ...")
        suffix = "_float32"

    tflite_model = converter.convert()

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = os.path.join(tflite_dir, f"waste_classifier{suffix}.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✓ TFLite model saved: {output_path} ({size_mb:.2f} MB)")

    # ── Quick verification ───────────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"  Input shape  : {input_details[0]['shape']}")
    logger.info(f"  Input dtype  : {input_details[0]['dtype']}")
    logger.info(f"  Output shape : {output_details[0]['shape']}")
    logger.info(f"  Output dtype : {output_details[0]['dtype']}")

    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Convert model to TFLite")
    parser.add_argument(
        "--quantize",
        choices=["float32", "float16", "int8"],
        default="float32",
        help="Quantisation method (default: float32)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model path")
    args = parser.parse_args()

    cfg = load_config()
    convert_to_tflite(cfg, model_path=args.model, quantize=args.quantize)
