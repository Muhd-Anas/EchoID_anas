# Voice Speaker Recognition - Model Training Module
"""
This module handles the training pipeline for the CNN speaker recognition model.
It integrates:
    - Dynamic model construction (via ModelBuilder)
    - Config-driven training parameters (batch size, epochs, validation split)
    - Automatic callback loading (early stopping, LR scheduling)
    - Version-controlled model saving

Behavior:
---------
- Loads model + callbacks based on config.
- Trains on mel-spectrogram features.
- Automatically bumps model version if a previous version already exists.
- Saves trained model inside `models/cnn_model_<version>/model_<version>.keras`.

Example:
--------
>>> from src.models.trainer import Trainer
>>> trainer = Trainer()
>>> model, history = trainer.train(x_train, y_train)

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Model Training & Versioned Model Saving
License: MIT
"""


import logging
from pathlib import Path
from src.utils import config_utils as cfg


# ------------------ Module Logger ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================
# Trainer Class
# =============================================================

class Trainer:
    """
    Trainer
    -------
    Manages the full model training workflow.

    Responsibilities:
    - Loads configuration settings.
    - Constructs the CNN model using ModelBuilder.
    - Loads and attaches callbacks.
    - Handles training execution and version-controlled saving.

    Attributes
    ----------
    config : dict
        Loaded configuration dictionary.
    model : keras.Model
        The compiled CNN model ready for training.
    callbacks : list
        Configured Keras callbacks (EarlyStopping, ReduceLROnPlateau).
    """

    def __init__(self):
        """Initialize Trainer by loading config and preparing model + callbacks."""
        try:
            self.config = cfg.read_config()
            version = self.config.get("version", "v1")
            logger.info(
                f"Initializing Model Training Process - Version: {version}")

            from .model_builder import ModelBuilder
            self.builder = ModelBuilder()
            self.model = self.builder.build_model()
            self.model.summary()

            from .callbacks import get_callbacks
            self.callbacks = get_callbacks()

            training_cfg = self.config.get("training", {})
            self.batch_size = training_cfg.get("batch_size", 32)
            self.epochs = training_cfg.get("epochs", 10)
            self.validation_split = training_cfg.get("validation_split", 0.2)
        except Exception as e:
            logger.error(
                f"❌ Trainer initialization failed: {e}", exc_info=True)
            raise

    # -------------------------- Training -------------------------------

    def train(self, x_train, y_train, verbose: int = 1, shuffle: bool = True):
        """
        Train the CNN model on mel-spectrogram input data.

        Args:
            x_train (np.ndarray): Training features of shape (N, 64, 188).
            y_train (np.ndarray): Training labels of shape (N,).
            verbose (int): Verbosity of Keras training logs.
            shuffle (bool): Whether to shuffle training data.

        Returns:
            tuple: (trained model, history object)
        """
        try:
            logger.debug("Preparing data for training...")
            x_mel_train = x_train.reshape(-1, 64, 188, 1)
            y_mel_train = y_train.reshape(-1, 1)

            logger.info("Training started...")
            history = self.model.fit(
                x=x_mel_train,
                y=y_mel_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=self.callbacks,
                verbose=verbose,
                shuffle=shuffle
            )

            logger.info("✅ Training completed successfully. Saving model...")
            self.__save_model()
            cfg.save_config()
            self.model.summary()

            return self.model, history

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}", exc_info=True)
            raise

    # -------------------------- Model Saving ---------------------------

    def __save_model(self):
        """
        Save the trained model with version control.
        If the target directory already exists → bump version automatically.
        """

        try:
            config = cfg.read_config()
            version = config.get("version", "v1")
            PROJECT_ROOT = Path(__file__).resolve().parent[3]

            save_dir_rel = config["paths"].get(
                "save_dir", f"models/cnn_model_{version}")
            save_dir = (PROJECT_ROOT / save_dir_rel).resolve()

            # If folder exists -> bump version
            if save_dir.exists():
                logger.debug(
                    f"⚠️ Model directory {save_dir} already exists. Bumping version...")
                version = cfg.bump_version()
                config = cfg.read_config()
                save_dir_rel = config["paths"].get(
                    "save_dir", f"models/cnn_model_{version}")
                save_dir = (PROJECT_ROOT / save_dir_rel).resolve()

            save_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"model_{version}.keras"
            save_path = save_dir / model_filename
            self.model.save(save_path)

            logger.info(f"✅ Model successfully saved → {save_path}")

        except Exception as e:
            logger.error(
                f"❌ Error during model saving setup: {e}", exc_info=True)
            raise


# ---------------------------------------------------------
# Run module independently (debug only)
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
