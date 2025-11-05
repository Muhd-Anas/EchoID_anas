# Voice Speaker Recognition - Mel-Spectrogram Convertion & Augmentation Module
"""
This module handles:
    1. Conversion of raw audio waveforms into mel-spectrograms.
    2. Feature-level augmentations such as:
        - Time & Frequency Masking (SpecAugment)
        - Vocal Tract Length Perturbation (VTLP)

It is designed to work on batched audio np-arrays and provide
robust feature diversity for training speaker recognition models.

Example:
    >>> from src.data.mel_processor import MelAugmentor, WaveformToMel
    >>> processor = WaveformToMel(sr=16000, n_mels=64)
    >>> mel = processor.waveform_to_mel(audio)
    >>> augmentor = MelAugmentor()
    >>> mel_aug, label_aug = augmentor.run(mel, label)

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Audio Augmentation
License: MIT
"""

import numpy as np
import librosa
import logging

# ------------------ Logging Configuration ------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ============================================================
# Mel-Spectrogram Augmentor Class
# ============================================================

class MelAugmentor:
    """
    Handles mel-spectrogram feature-space augmentations such as
    SpecAugment (time/frequency masking) and VTLP (Vocal Tract Length Perturbation).

    This class expects mel-spectrogram tensors, not raw audio. It performs
    feature-space augmentation to improve model robustness.

    Example
    -------
    >>> import numpy as np
    >>> from src.data.mel_processor import MelAugmentor
    >>>
    >>> # Initialize the augmentor
    >>> augmentor = MelAugmentor(sr=16000, n_mels=64)
    >>>
    >>> # Run augmentation
    >>> mel_aug, labels_aug = augmentor.run(mel_specs, labels, num_aug=2, shuffle=True)
    >>>
    >>> # Check shapes after augmentation
    >>> print(mel_aug.shape, labels_aug.shape)
    (80, 32, 64, 188, 1) (80, 32)
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256
    ):
        """
        Handles mel-spectrogram feature-space augmentations such as
        SpecAugment (time/frequency masking) and VTLP (Vocal Tract Length Perturbation).

        Args:
            sr (int): Sampling rate of the audio.
            n_mels (int): Number of mel-frequency bins.
            n_fft (int): FFT window size.
            hop_length (int): Hop length for STFT computation.
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    # -------- SpecAugment - Time and Frequency Masking ------------

    def _time_frequency_masking(self, mel, time_mask_param=(5, 20), freq_mask_param=(3, 12)):
        """
        Apply SpecAugment-style time and frequency masking on the mel-spectrogram.

        Args:
            mel (np.ndarray): Input mel-spectrogram of shape (n_mels, time, 1) or (n_mels, time).
            time_mask_param (tuple): Range for time mask width.
            freq_mask_param (tuple): Range for frequency mask width.

        Returns:
            np.ndarray: Augmented mel-spectrogram.
        """
        augmented = mel.copy()
        num_mels, num_frames = augmented.shape[:2]

        # Randomly choose augmentation type
        choice = np.random.rand()

        # Time Masking
        if choice < 0.44:
            t = np.random.randint(*time_mask_param)
            t0 = np.random.randint(0, max(1, num_frames - t))
            augmented[:, t0:t0 + t] = 0

        # Frequency Masking
        elif choice < 0.55:
            f = np.random.randint(*freq_mask_param)
            f0 = np.random.randint(0, max(1, num_mels - f))
            augmented[f0:f0 + f, :] = 0

        # Both
        else:
            t = np.random.randint(*time_mask_param)
            t0 = np.random.randint(0, max(1, num_frames - t))
            augmented[:, t0:t0 + t] = 0
            f = np.random.randint(*freq_mask_param)
            f0 = np.random.randint(0, max(1, num_mels - f))
            augmented[f0:f0 + f, :] = 0

        return augmented

    # -------- Vocal Tract Length Perturbation (VTLP) --------------

    def _vtlp(self, mel, alpha_range=(0.9, 1.1)):
        """
        Apply Vocal Tract Length Perturbation to simulate speaker variability.

        Args:
            mel (np.ndarray): Input mel-spectrogram.
            alpha_range (tuple): Range for the warping factor.

        Returns:
            np.ndarray: Warped mel-spectrogram.
        """
        alpha = np.random.uniform(*alpha_range)
        warped_mel = np.zeros_like(mel)
        num_freq_bins = mel.shape[0]
        f0 = int(num_freq_bins * 0.4)

        for i in range(num_freq_bins):
            if i <= f0:
                warped_i = int(alpha * i)
            else:
                warped_i = int(
                    (num_freq_bins - alpha * f0) /
                    (num_freq_bins - f0) * (i - f0) + alpha * f0
                )
            warped_i = min(warped_i, num_freq_bins - 1)
            warped_mel[i, :] = mel[warped_i, :]

        return warped_mel

    # ----------- Random Augmentation Pipeline --------------------

    def _random_augment(self, mel_batch: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to each mel-spectrogram in a batch.

        Args:
            mel_batch (np.ndarray): Batch of mel-spectrograms with shape (batch, n_mels, time, 1)

        Returns:
            np.ndarray: Augmented mel-spectrogram batch.
        """
        augmented_batch = []
        augmentations = [
            ("masking", 0.5, self._time_frequency_masking),
            ("vtlp", 0.5, self._vtlp),
        ]

        for i, mel in enumerate(mel_batch):
            augmented = mel.squeeze().copy()
            for name, p, func in augmentations:
                if np.random.random() < p:
                    try:
                        augmented = func(augmented)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Augmentation '{name}' failed for sample {i}: {e}")

            # Restore 3D shape (n_mels, time, 1)
            if augmented.ndim == 2:
                augmented = augmented[..., np.newaxis]

            augmented_batch.append(augmented)

        return np.stack(augmented_batch, axis=0)

    # ----------- Full Augmentation Runner -------------------------

    def run(self, mel: np.ndarray, label: np.ndarray, num_aug: int = 1, shuffle: bool = True):
        """
        Run mel augmentation pipeline.

        Args:
            mel (np.ndarray): Original mel-spectrogram batchs
            label (np.ndarray): Corresponding labels.
            num_aug (int): Number of augmentation versions to generate.
            shuffle (bool): Whether to shuffle the final dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: (augmented_mel, augmented_labels)
        """
        logger.info("🚀 Starting mel-spectrogram augmentation pipeline...")

        augmented = []

        all_labels = np.tile(label, (num_aug + 1, 1))

        for i in range(num_aug):
            logger.info(f"🔁 Running augmentation round {i + 1}/{num_aug}...")
            try:
                augment = np.stack([
                    self._random_augment(mel[j])
                    for j in range(mel.shape[0])
                ])
                augmented.append(augment)
            except Exception as e:
                logger.error(f"❌ Augmentation for mel-spectrogram failed: {e}")

        # Combine original + augmented
        augment = np.stack(augmented).reshape(-1, *mel.shape[1:])
        combined = np.concatenate((mel, augment), axis=0)

        if shuffle:
            indices = np.arange(combined.shape[0])
            np.random.shuffle(indices)
            combined = combined[indices]
            all_labels = all_labels[indices]
            logger.info("🔀 Shuffled augmented dataset.")

        logger.info(f"✅ Augmentation complete! Final shape: {combined.shape}")
        return combined, all_labels


# ============================================================
# Waveform → Mel-Spectrogram Converter
# ============================================================

class WaveformToMel:
    """
    Converts raw audio waveforms into normalized mel-spectrograms.

    This class takes raw waveforms (e.g., loaded from TensorFlow, Librosa, or
    other audio loaders) and converts them to mel-spectrograms in decibel scale.
    It also supports conversion of multiple batches of waveforms.

    Example
    -------
    >>> import numpy as np
    >>> from src.data.mel_processor import WaveformToMel
    >>>
    >>> # Initialize the converter
    >>> converter = WaveformToMel(sr=16000, n_mels=64)
    >>>
    >>> # Convert all batches to mel-spectrograms
    >>> mel_specs, labels = converter.run(x_train, y_train)
    >>>
    >>> # mel_specs is a 4D array (batch, sample, n_mels, time, 1)
    >>> print(mel_specs.shape)
    (2, 32, 64, 188, 1)
    """

    def __init__(
            self,
            sr: int = 16000,
            n_mels: int = 64,
            n_fft: int = 1024,
            hop_length: int = 256
    ):
        """
        Converts waveform batches into mel-spectrograms using librosa.

        Args:
            sr (int): Sampling rate.
            n_mels (int): Number of mel-frequency bins.
            n_fft (int): FFT window size.
            hop_length (int): Hop length between frames.
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def waveform_to_mel(self, waveform: np.ndarray):
        """
        Convert a single waveform batch into mel-spectrograms.

        Args:
            waveform (np.ndarray): Batch of shape (batch_size, samples, 1)

        Returns:
            np.ndarray: Normalized mel-spectrograms
        """
        mel_specs = []

        for i in range(waveform.shape[0]):
            try:
                sample = waveform[i].squeeze()
                mel_spec = librosa.feature.melspectrogram(
                    y=sample,
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mel_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_specs.append(mel_db)
            except Exception as e:
                logger.error(
                    f"❌ Failed to convert sample {i} to mel-spectrogram: {e}")

        mel_specs = np.array(mel_specs)[..., np.newaxis]
        mel_specs = (mel_specs - mel_specs.min()) / \
            (mel_specs.max() - mel_specs.min() + 1e-9)
        return mel_specs

    def run(self, audio, label):
        """
        Convert multiple waveform batches into mel-spectrograms.

        Args:
            audio (np.ndarray): Audio batches of shape (num_batches, batch_size, samples, 1)
            label (np.ndarray): Corresponding labels.

        Returns:
            tuple[np.ndarray, np.ndarray]: (mel_spectrograms, labels)
        """
        logger.info("🎧 Converting waveform batches to mel-spectrograms...")
        mel_batches = []

        for i in range(audio.shape[0]):
            mel = self.waveform_to_mel(audio[i])
            mel_batches.append(mel)
            logger.info(f"Converted batch {i + 1}/{len(audio)}")

        mel_batches = np.array(mel_batches)
        logger.info(
            f"✅ Mel conversion completed. Final shape: {mel_batches.shape}")
        return mel_batches, label


# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
