# Voice Speaker Recognition - Audio Dataset Loader Module
"""
This module handles the loading and preprocessing of audio files 
for the Voice Speaker Recognition CNN project.

It uses TensorFlow/Keras' `audio_dataset_from_directory` to load WAV files,
convert them into TensorFlow datasets, and then prepares them as NumPy arrays 
for further augmentation and model training.

Dataset Structure:
------------------
data/
  ├── speaker0/
  │    ├── audio_1.wav
  │    ├── audio_2.wav
  │    └── ...
  ├── speaker1/
  │    ├── audio_1.wav
  │    ├── audio_2.wav
  │    └── ...

Notes:
------
- The dataset is automatically split into training and testing subsets (default 80/20).
- This script only loads and prepares raw audio data. 
  Use other modules for augmentation and mel-spectrogram feature extraction.
- logger is used instead of print statements for cleaner debugging.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Audio Dataset Loader
License: MIT
"""


from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os

# ------------------ TensorFlow Environment Config ------------------
# Suppress TensorFlow C++ logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable oneDNN optimization info (optional, "0" for off)
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ------------------ Module Logger ------------------
logger = logging.getLogger(__name__)  # <— module-specific logger
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AudioDatasetLoader:
    """
    AudioDatasetLoader
    ------------------
    A utility class to load and prepare audio datasets from directory structures
    for training machine learning or deep learning models.

    It uses Keras’ built-in `audio_dataset_from_directory()` to load audio files
    along with their inferred labels. It then converts the dataset to NumPy arrays
    and splits it into training and testing sets for model input.

    Typical Workflow
    ----------------
    1. Initialize the class with dataset directory and configuration.
    2. Call `load_dataset()` to automatically load, preprocess, and split data.
    3. Use the returned arrays (`x_train`, `x_test`, `y_train`, `y_test`)
     for model training and evaluation.

    Attributes
    ----------
    file_path : str
        Root path of dataset containing subdirectories per class label
    sr : int
        Target sampling rate for audio files
    batch_size : int
        Number of audio samples per batch
    seed : int
        Random seed for reproducibility   

    Example
    -------
    >>> from dataset_loader import AudioDatasetLoader
    >>>
    >>> loader = AudioDatasetLoader(
    ...     file_path="C:/voice-speaker-binary-classifier/data/",
    ...     sr=16000,
    ...     batch_size=32
    ... )
    >>> x_train, x_test, y_train, y_test = loader.load_dataset()
    >>>
    >>> print(x_train.shape, y_train.shape)
    (20, 32, 48000, 1) (20, 32)
    """

    def __init__(self, file_path: str, sr: int = 16000, batch_size: int = 32, seed: int = 1337):
        self.file_path = file_path
        self.sr = sr
        self.batch_size = batch_size
        self.seed = seed
        logger.info(
            f"Initialized AudioDatasetLoader with path={file_path}, sr={sr}, batch_size={batch_size}")

    # ---------------------------------------------------------
    def load_dataset(self):
        """
        Load the dataset from the given directory using Keras utilities.

        Returns:
            tuple: Train-test split of (x_train, x_test, y_train, y_test)
        """

        logger.info(
            "Loading dataset from directory... this may take a few seconds.")

        import keras

        dataset = keras.utils.audio_dataset_from_directory(
            directory=self.file_path,
            labels="inferred",            # Labels are inferred from subdirectory names
            label_mode="int",             # Labels are returned as integers
            batch_size=self.batch_size,
            shuffle=True,                 # Shuffle to improve training performance
            # Truncate or pad each audio to 3s (48000 samples @ 16kHz)
            output_sequence_length=48000,
            seed=self.seed
        )

        logger.info(
            "Dataset loaded successfully. Converting to NumPy arrays...")
        return self._dataset_to_numpy(dataset)

    # ---------------------------------------------------------
    def _dataset_to_numpy(self, dataset):
        """
        Convert TensorFlow dataset to NumPy arrays for easier manipulation.

        Args:
            dataset: A TensorFlow `_BatchDataset` containing (audio, label) pairs.

        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        x_list, y_list = [], []

        for i, (audio, label) in enumerate(dataset.as_numpy_iterator()):
            logger.debug(f"Processing batch {i+1}")
            x_list.append(audio)
            y_list.append(label)

        logger.info(
            f"Collected {len(x_list)} batches. Concatenating into full arrays...")

        x_data = np.concatenate(x_list)
        y_data = np.concatenate(y_list)

        num_batches = x_data.shape[0] // self.batch_size

        logger.info(
            f"Converting to {num_batches} batches with batch_size of {self.batch_size}")

        # Trim data so it divides evenly
        x_data_trimmed = x_data[: num_batches * self.batch_size]
        y_data_trimmed = y_data[: num_batches * self.batch_size]
        x_data_batch = x_data_trimmed.reshape(
            num_batches, self.batch_size, x_data.shape[-2], 1)
        y_data_batch = y_data_trimmed.reshape(num_batches, self.batch_size)

        logger.info(
            f"Final dataset shape: X={x_data_batch.shape}, Y={y_data_batch.shape}")
        return self._train_test_split(x_data_batch, y_data_batch)

    # ---------------------------------------------------------
    def _train_test_split(self, audio, label, random_state: int = 43):
        """
        Split dataset into training and testing sets.

        Args:
            audio (np.ndarray): Audio waveform data.
            label (np.ndarray): Corresponding labels.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        logger.info("Splitting dataset into train and test sets (80/20)...")

        x_train, x_test, y_train, y_test = train_test_split(
            audio, label, train_size=0.8, random_state=random_state
        )

        logger.info(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
        logger.info(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
        return x_train, x_test, y_train, y_test


# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if "__main__" == __name__:
    ...
