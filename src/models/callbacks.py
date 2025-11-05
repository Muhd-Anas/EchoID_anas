# Voice Speaker Recognition - Training Callbacks Module
"""
This module constructs a set of Keras callbacks used during training,
configured dynamically from `config.yaml`.

Callbacks Included:
-------------------
- EarlyStopping: Stops training when validation performance stops improving.
- ReduceLROnPlateau: Reduces learning rate when validation performance stagnates.

This ensures consistent, configurable training behavior across model versions.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Training Utilities (Callbacks)
License: MIT
"""



import os
import logging
from pathlib import Path
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils.config_utils import read_config


# ------------------ Module Logger ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================
# Callback Factory Function
# =============================================================

def get_callbacks():
    """
    Create a list of Keras callbacks based on `config.yaml`.

    Returns:
        list: A list of configured Keras callback objects.

    Raises:
        RuntimeError: If configuration is invalid or missing.
    """
    try:
        config = read_config()
        training_cfg = config.get("training", {})
        early_cfg = training_cfg.get("early_stopping", {})
        lr_cfg = training_cfg.get("reduce_lr", {})
        version = config.get("version", "v1")

        callbacks = []

        # ----------------- Early Stopping -----------------
        early_stop = EarlyStopping(
            monitor=early_cfg.get("monitor", "val_loss"),
            patience=early_cfg.get("patience", 5),
            restore_best_weights=early_cfg.get("restore_best_weights", True),
            start_from_epoch=early_cfg.get("start_from_epoch", 0),
            verbose=1
        )
        callbacks.append(early_stop)

        # ----------------- Learning Rate Scheduler -----------------
        reduce_lr = ReduceLROnPlateau(
            monitor=lr_cfg.get("monitor", "val_loss"),
            factor=lr_cfg.get("factor", 0.5),
            patience=lr_cfg.get("patience", 3),
            min_lr=lr_cfg.get("min_lr", 1e-6),
            verbose=1
        )
        callbacks.append(reduce_lr) 

        logger.debug(f"Configured Callbacks: {callbacks}")
        return callbacks

    except Exception as e:
        logger.error(f"❌ Failed to read training configuration: {e}", exc_info=True)
        raise RuntimeError("Callback initialization failed.") from e
    

# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
