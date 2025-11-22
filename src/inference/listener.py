# Voice Speaker Recognition - Inference Module
"""
This module handles the real-time inference for voice speaker recognition.
It loads the trained CNN speaker recognition model and provides a GUI interface
for recording audio and performing speaker inference.

Behavior:
---------
- Loads configuration to determine model paths and audio settings.
- Automatically locates the specific versioned model file.
- Provides a threaded recording interface to prevent UI freezing.
- Displays real-time prediction results based on confidence thresholds.

Example:
--------
>>> from src.inference.inference_engine import InferenceApp
>>> app = InferenceApp()
>>> app.run()

Name: EchoID    
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Real-time Inference & GUI
License: MIT
"""


import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import logging
import librosa
import os
from pathlib import Path
from vad import EnergyVAD
from src.utils import config_utils as cfg

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
# Inference Application Class
# =============================================================

class InferenceApp:
    """
    InferenceApp
    ------------
    Manages the GUI and logic for real-time speaker recognition.

    Responsibilities:
    - Initializes the UI window.
    - Loads the pre-trained CNN model from disk based on config.
    - Handles audio recording in a separate thread.
    - Performs preprocessing and model prediction.

    Attributes
    ----------
    config : dict
        Loaded configuration dictionary.
    model : keras.Model
        The loaded voice recognition model.
    confidence_threshold : float
        The probability threshold for positive identification.
    sample_rate : int
        Audio sampling rate (default: 16000).
    duration : int
        Recording duration in seconds.

    Example:
    --------
    >>> from src.inference.inference_engine import InferenceApp
    >>> app = InferenceApp()
    >>> app.run()    
    """

    def __init__(self):
        """
        Initialize the Inference Application.
        Loads config, model, and builds the GUI.
        """
        try:
            self.config = cfg.read_config()
            self.version = self.config.get("version", "v1")

            logger.info(
                f"Initializing Inference Module - Version: {self.version}")

            # Config parameters
            inference_cfg = self.config.get("inference", {})
            self.confidence_threshold = inference_cfg.get(
                "confidence_threshold", 0.7)

            # Audio defaults (fallback if not in config)
            self.sample_rate = inference_cfg.get("sample_rate", 16000)
            self.duration = inference_cfg.get("duration", 3)

            # State variables
            self.is_recording = False
            self.audio_data = None
            self.master = tk.Tk()

            # Load Model
            self.model = self.__load_model()

            # Build UI
            self.__init_ui()

        except Exception as e:
            logger.error(
                f"❌ Inference App initialization failed: {e}", exc_info=True)
            raise

    # -------------------------- Initialization Helpers ---------------------------

    def __load_model(self):
        """
        Locate and load the trained Keras model based on version config.

        Returns:
            keras.Model: The loaded CNN model.

        Raises:
            FileNotFoundError: If the model directory or file does not exist.
        """
        try:
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            save_dir_rel = self.config["paths"].get(
                "save_dir", f"models/cnn_model_{self.version}"
            )
            save_dir = (PROJECT_ROOT / save_dir_rel).resolve()

            if not save_dir.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {save_dir}")

            model_filename = f"model_{self.version}.keras"
            load_path = save_dir / model_filename

            if not load_path.exists():
                raise FileNotFoundError(
                    f"Keras model file not found at: {load_path}")

            logger.info(
                f"Loading model from: {load_path}. This may take a few seconds...")

            # Local import to avoid overhead if not needed immediately
            import keras
            return keras.saving.load_model(load_path)

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}", exc_info=True)
            raise

    def __init_ui(self):
        """Set up the Tkinter GUI components."""
        try:
            self.master.geometry("400x200")
            self.master.configure(bg="#181A1A")
            self.master.title(f"EchoID - Inference {self.version}")

            # Status Label
            self.status_label = tk.Label(
                self.master,
                text="Press mic to record.",
                fg="#39F8D8",
                bg="#181A1A",
                font=("Arial", 10)
            )
            self.status_label.pack(pady=(20, 15))

            # Mic Button
            self.mic_button = tk.Button(
                self.master,
                bg="#0c92cf",
                fg="#ffffff",
                width=10,
                height=1,
                font=("Arial", 10),
                text="Start Mic",
                command=self.toggle_recording
            )
            self.mic_button.pack(pady=15)

            # Output Label
            self.output_label = tk.Label(
                self.master,
                text="",
                bg="#181A1A",
                fg="#39F8D8",
                font=("Arial", 12, "bold")
            )
            self.output_label.pack(pady=(15, 1))

        except Exception as e:
            logger.error(f"❌ GUI Setup failed: {e}", exc_info=True)
            raise

    # -------------------------- Core Logic ---------------------------

    def toggle_recording(self):
        """
        Handle the Start/Stop microphone button click.
        Launches recording in a separate thread to avoid blocking the UI.
        """
        self.output_label.config(text="")

        if not self.is_recording:
            self.is_recording = True
            self.mic_button.config(text="Recording...",
                                   state="disabled", relief="sunken")
            threading.Thread(target=self._record_routine, daemon=True).start()
        else:
            # This branch is theoretically unreachable if button is disabled during record,
            # but kept for logic completeness if manual stop is implemented.
            self.is_recording = False
            self.mic_button.config(text="Start Mic")

    def _record_routine(self):
        """
        Internal method to record audio via sounddevice.
        Executed in a background thread.
        """
        try:
            logger.debug("Recording started...")
            self.master.after(
                0, lambda: self.status_label.config(text="Recording..."))

            # Record
            self.audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            logger.debug("Recording finished.")
            self.master.after(
                0, lambda: self.status_label.config(text="Processing..."))

            # Trigger Inference
            self._inference_routine()

        except Exception as e:
            logger.error(f"❌ Error during recording: {e}", exc_info=True)
            self.master.after(1, lambda: self.status_label.config(
                text="Error in recording"))

        finally:
            # Reset UI state
            self.is_recording = False
            self.master.after(1, lambda: self.mic_button.config(
                text="Start Mic", state="normal", relief="raised"))

    def _inference_routine(self):
        """
        Process recorded audio, check VAD, and run model inference.
        Updates the GUI with the result.
        """
        try:
            # 1. Prepare Data
            data = np.array(self.audio_data).reshape(-1)

            data = librosa.resample(
                data, orig_sr=self.sample_rate, target_sr=16000)

            # Pad if too short
            required_length = self.sample_rate * self.duration
            if len(data) < required_length:
                padding = required_length - len(data)
                data = np.pad(data, (0, padding), mode='constant')
            else:
                data = data[:required_length]

            # 2. Voice Activity Detection (VAD)
            vad = EnergyVAD(
                sample_rate=self.sample_rate,
                frame_length=20,
                frame_shift=10,
                energy_threshold=0.03
            )
            voice_activity = vad(data)

            # If strictly no voice is detected
            if not voice_activity.any():
                logger.debug("VAD: No speech detected.")
                self.master.after(1, lambda: self.status_label.config(
                    text="Press mic to record."))
                self.master.after(1, lambda: self.output_label.config(
                    text="No speech detected", fg="orange"))
                return

            # 3. Preprocessing (Mel Spectrogram)
            # Ensure these parameters match your Training parameters exactly
            mel_db = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=data, sr=self.sample_rate, n_mels=64, n_fft=1024, hop_length=256
                ),
                ref=np.max
            )

            # Normalize
            mel_norm = (mel_db - mel_db.min()) / \
                (mel_db.max() - mel_db.min() + 1e-9)
            # Shape: (1, 64, Time(188), 1)
            mel_norm = mel_norm[np.newaxis, ..., np.newaxis]

            logger.debug(f"Input shape for model: {mel_norm.shape}")

            # 4. Prediction
            prediction = self.model.predict(mel_norm, verbose=0)
            confidence = prediction[0][0] if prediction.ndim == 2 else prediction[0]

            logger.debug(f"Prediction Confidence: {confidence:.4f}")

            # 5. UI Update
            if prediction > self.confidence_threshold:
                result_text = f"Speaker Authenticated ({confidence:.2f})"
                color = "#00FF00"  # Green
            else:
                result_text = f"Other Speaker Detected ({confidence:.2f})"
                color = "#FF0000"  # Red

            self.master.after(0, lambda: self.status_label.config(
                text="Press mic to record."))
            self.master.after(0, lambda: self.output_label.config(
                text=result_text, fg=color))

        except Exception as e:
            logger.error(f"❌ Error during inference: {e}", exc_info=True)
            self.master.after(0, lambda: self.output_label.config(
                text="Inference Error", fg="red"))

    def run(self):
        """Start the main GUI event loop."""
        try:
            logger.info("Starting GUI Mainloop...")
            self.master.mainloop()
        except KeyboardInterrupt:
            logger.info("Application stopped by user.")
        except Exception as e:
            logger.error(f"❌ Application crashed: {e}", exc_info=True)


# ---------------------------------------------------------
# Run module independently (debug only)
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
