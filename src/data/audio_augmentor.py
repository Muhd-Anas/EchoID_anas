# Voice Speaker Recognition - Audio Augmentation Module
"""
This module provides beginner-friendly functions and a class for batch-wise
audio data augmentation. It supports randomizing each batch, processing datasets
in batches, and combining original and augmented data for model training.

Usage Example:
--------------
    from audio_augmentation import AudioPreprocessing

    augmentor = AudioAugmentation(sr=16000, batch_size=32)
    x_aug, y_aug = augmentor.run(audio_batch, labels, num_aug=2, shuffle=True)

Notes:
------
- Each augmentation is applied randomly with defined probabilities.
- Extendable for time/frequency masking, VTLP, or spectrogram augmentations.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Audio Augmentation
License: MIT
"""

import os
import logging
import numpy as np
import librosa


# ------------------ Logging Configuration ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ------------------ Audio Augmentation Class ------------------

class AudioAugmentor:
    """
    AudioPreprocessing
    ------------------
    A class to perform audio data augmentation for sound datasets.

    This class enhances dataset diversity by applying random audio transformations
    such as adding Gaussian noise, pitch shifting, and amplitude scaling.
    These augmentations help improve model generalization during training.

    Typical Workflow
    ----------------
    1. Initialize the class with your preferred settings.
    2. Pass your audio batches (NumPy arrays) and labels to the `run()` method.
    3. Receive augmented + original audio data and labels ready for model training.

    Attributes
    ----------
    sr : int
        Sampling rate of audio (e.g., 16000 Hz)
    batch_size : int
        Number of audio samples processed per batch

    Example
    -------
    >>> import numpy as np
    >>> from audio_augmentation import AudioPreprocessing
    >>>
    >>> augmentor = AudioPreprocessing(sr=16000, batch_size=32)
    >>> x_aug, y_aug = augmentor.run(audio, labels, num_aug=2, shuffle=True)
    >>>
    >>> print(x_aug.shape, y_aug.shape)
    (40, 32, 48000, 1), (40, 32)
    """

    def __init__(self, sr: int = 16000,  batch_size: int = 32):
        """
        Initialize the audio augmentor with key parameters.

        Args:
            sr (int): Sampling rate of audio (e.g., 16000 Hz)
            batch_size (int): Number of audio samples per batch
        """
        self.sr = sr  # Save the sample rate for later use
        self.batch_size = batch_size  # Save batch size
        logger.info(
            f"Audio Augmentation initialized with sr={sr}, batch_size={batch_size}")

    # ------------------ Augmentation Methods ------------------

    def _add_noise(self, audio: np.ndarray, noise_factor: float = None) -> np.ndarray:
        """
        Add random Gaussian noise to the input audio.

        Parameters
        ----------
        audio : np.ndarray
            Original audio waveform (batch, samples, 1)
        noise_factor : float, optional
            Intensity of the noise; random between 0.001-0.01 if not given.

        Returns
        -------
        np.ndarray
            Noisy audio waveform clipped to [-1, 1]
        """

        try:
            noise_factor = noise_factor or np.random.uniform(0.001, 0.01)
            noise = np.random.randn(*audio.shape)
            augmented = np.clip(audio + noise_factor * noise, -1.0, 1.0)
            logger.debug(f"Applied Gaussian noise (factor={noise_factor:.5f})")
            return augmented
        except Exception as e:
            logger.error(f"Noise augmentation failed: {e}")
            return audio

    def _pitch_shift(self, audio: np.ndarray, n_steps=None) -> np.ndarray:
        """
        Randomly shift the pitch of each audio sample in a batch.

        Parameters
        ----------
        audio : np.ndarray
            Audio waveform (batch, samples, 1)
        n_steps : int or np.ndarray, optional
            Number of semitones to shift; random between -3 and 3 if None.

        Returns
        -------
        np.ndarray
            Pitch-shifted audio with same shape as input
        """

        try:
            if n_steps is None:
                n_steps = np.random.randint(-3, 4, size=audio.shape[0])
            shifted = np.stack([
                librosa.effects.pitch_shift(
                    sample.flatten(), sr=self.sr, n_steps=int(step))
                for sample, step in zip(audio, n_steps)
            ])
            logger.debug(f"Applied pitch shift with random steps.")
            return shifted
        except Exception as e:
            logger.error(f"Pitch shift augmentation failed: {e}")
            return audio

    def _amplitude_scaling(self, audio: np.ndarray, scale=None) -> np.ndarray:
        """
        Randomly scale the amplitude (volume) of audio.

        Parameters
        ----------
        audio : np.ndarray
            Audio waveform (batch, samples, 1)
        scale : float or np.ndarray, optional
            Amplitude scale factor; random 0.6-1.2 if None.

        Returns
        -------
        np.ndarray
            Amplitude-scaled audio waveform
        """

        try:
            scale = scale or np.random.uniform(0.6, 1.2, size=audio.shape[0])
            scaled = np.stack([
                np.clip(sample * factor, -1.0, 1.0)
                for sample, factor in zip(audio, scale)
            ])
            logger.debug(f"Applied amplitude scaling.")
            return scaled
        except Exception as e:
            logger.error(f"Amplitude scaling failed: {e}")
            return audio

    def _random_augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply a random combination of augmentations to the input batch.

        Parameters
        ----------
        audio : np.ndarray
            Original batch of audio data

        Returns
        -------
        np.ndarray
            Augmented batch of audio data
        """

        augmented = audio.copy()
        augmentations = [
            ("noise", 0.5, self._add_noise),
            ("pitch", 0.5, self._pitch_shift),
            ("amplitude", 0.5, self._amplitude_scaling)
        ]

        # Randomly apply augmentations based on probabilities
        for name, prob, func in augmentations:
            if np.random.random() < prob:
                try:
                    augmented = func(augmented)
                    logger.debug(f"Applied {name} augmentation")
                except Exception as e:
                    logger.warning(f"{name} augmentation failed: {e}")

        return augmented

    # ------------------ Main Augmentation Runner ------------------

    def run(self, audio: np.ndarray, label: np.ndarray,
            num_aug: int = 1, shuffle: bool = True):
        """
        Execute the full augmentation pipeline on input batches.

        Parameters
        ----------
        audio : np.ndarray
            Original audio data (batch, samples, 1)
        label : np.ndarray
            Labels corresponding to the audio samples
        num_aug : int
            Number of augmentation rounds to perform
        shuffle : bool
            Whether to shuffle the combined dataset

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Augmented (and optionally shuffled) audio and labels
        """

        try:
            logger.info(
                f"🚀 Starting audio augmentation | num_aug={num_aug}, shuffle={shuffle}")
            logger.debug(f"Input shape: {audio.shape}")

            all_labels = np.tile(label, (num_aug + 1, 1))
            augment_batches = []

            for i in range(num_aug):
                logger.info(f"🔁 Running augmentation round {i + 1}/{num_aug}")
                aug = np.stack([
                    self._random_augment(audio[j]).reshape(
                        audio.shape[1], audio.shape[2]
                    )
                    for j in range(audio.shape[0])
                ])
                augment_batches.append(aug)

            augmented = np.stack(augment_batches).reshape(-1, *audio.shape[1:])
            combined_audio = np.concatenate((audio, augmented), axis=0)

            if shuffle:
                indices = np.arange(combined_audio.shape[0])
                np.random.shuffle(indices)
                combined_audio = combined_audio[indices]
                all_labels = all_labels[indices]
                logger.info("🔀 Shuffled augmented dataset")

            logger.info(
                f"✅ Augmentation complete. Final dataset shape: {combined_audio.shape}, {all_labels.shape}")
            return combined_audio, all_labels

        except Exception as e:
            logger.error(f"Audio augmentation failed: {e}")
            raise e


# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
