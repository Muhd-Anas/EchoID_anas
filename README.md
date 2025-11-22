# 🎙️ EchoID - Deep Voice Speaker Recognition System

![status: development](https://img.shields.io/badge/status-development-yellow?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Keras](https://img.shields.io/badge/keras-3.11.3-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **📚 Educational Purpose Project for Beginners**  
> EchoID is a project that started for the purpose of education for beginners who want to create voice recognition model but couldn't or don't want to write big boilerplate codes and don't have lots of data (so we need to do augmentation which will increase the code itself). This educational purpose project features **binary classification** for speaker recognition.

> **⚠️ IMPORTANT: This is an educational purpose project - you can't use it in production.**

**🎬 Demo Video Coming Soon!** A comprehensive walkthrough showing how to use this project will be available shortly.

---

## 📑 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Audio Chunking](#2-audio-chunking)
  - [3. Training the Model](#3-training-the-model)
  - [4. Real-Time Inference](#4-real-time-inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Future Roadmap](#future-roadmap)
- [License](#license)
- [Contact](#contact)

---

## 🎯 About the Project

**EchoID** is an educational deep learning project designed to help beginners learn voice speaker recognition. This project was created for those who want to build a voice recognition model without writing extensive boilerplate code or having large datasets (data augmentation is built-in to help with limited data). It leverages Convolutional Neural Networks (CNNs) trained on mel-spectrogram representations of audio signals to achieve accurate **binary classification** - distinguishing between a target speaker and others.

**This is purely for educational purposes and should not be used in production environments.**

### Key Highlights:
- 🧠 **Deep CNN Architecture**: Custom-built CNN model optimized for speaker recognition
- 🎵 **Mel-Spectrogram Features**: Converts raw audio to frequency-based representations
- 🔊 **Advanced Augmentation**: Multi-level augmentation (waveform + spectrogram) for robust learning
- 📊 **Config-Driven**: Fully configurable via YAML for easy experimentation
- 🎤 **Real-Time Inference**: GUI-based live speaker recognition
- 📈 **Metrics Tracking**: Comprehensive evaluation with accuracy, precision, recall, F1-score, and ROC-AUC

---

## ✨ Features

### Data Processing Pipeline
- **🎼 Audio Chunking**: Split long recordings into uniform 3-second segments
- **📚 Dataset Loading**: Efficient batch-wise audio loading with automatic train-test split (80/20)
- **🔄 Waveform Augmentation**: 
  - Gaussian noise injection
  - Pitch shifting (-3 to +3 semitones)
  - Amplitude scaling (0.6x to 1.2x)
- **🎶 Mel-Spectrogram Conversion**: Transform waveforms to 64x188 mel-spectrograms
- **🎨 Mel-Level Augmentation**: SpecAugment and VTLP for enhanced generalization

### Model Training
- **🏗️ Dynamic Model Builder**: Config-driven CNN architecture construction
- **⚡ Smart Callbacks**: 
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau
- **📊 Comprehensive Metrics**: Track accuracy, precision, recall, F1-score, and ROC-AUC
- **💾 Version Control**: Automatic model versioning and checkpointing

### Inference
- **🎯 Real-Time Recognition**: Live audio recording and prediction via GUI
- **🔇 Voice Activity Detection (VAD)**: Energy-based VAD for robust inference
- **⚙️ Configurable Thresholds**: Adjust confidence levels for predictions

---

## 🔍 How It Works

### Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION                                             │
│     └─ Collect audio samples into data/speaker0 & data/speaker1│
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  2. AUDIO CHUNKING (AudioChunker)                               │
│     └─ Split long recordings into 3-second WAV chunks           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  3. DATASET LOADING (AudioDatasetLoader)                        │
│     └─ Load audio files and create train/test split (80/20)    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  4. WAVEFORM AUGMENTATION (AudioAugmentor)                      │
│     ├─ Add Gaussian noise (0.001-0.01 factor)                  │
│     ├─ Pitch shift (±3 semitones)                              │
│     └─ Amplitude scaling (0.6x-1.2x)                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  5. MEL-SPECTROGRAM CONVERSION (WaveformToMel)                  │
│     └─ Convert waveforms to 64x188 mel-spectrograms            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  6. MEL AUGMENTATION (MelAugmentor)                             │
│     ├─ SpecAugment (time & frequency masking)                  │
│     └─ VTLP (Vocal Tract Length Perturbation)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  7. CNN TRAINING (Trainer)                                      │
│     ├─ Build CNN from config (3 Conv2D layers: 32→64→128)      │
│     ├─ Train with Early Stopping & LR Reduction                │
│     └─ Track metrics: Accuracy, Precision, Recall, F1, ROC-AUC │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  8. MODEL EVALUATION                                            │
│     └─ Evaluate on test set and generate performance reports   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  9. REAL-TIME INFERENCE (InferenceApp)                          │
│     └─ GUI-based live speaker recognition with VAD             │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Details

**Input Shape**: `(64, 188, 1)` - Mel-spectrograms normalized to [0, 1]  
**CNN Architecture**: 3 convolutional blocks with progressive filters (32 → 64 → 128)  
**Training**: Binary cross-entropy loss with Adam optimizer  
**Sample Rate**: 16 kHz  
**Chunk Duration**: 3 seconds (48,000 samples @ 16kHz)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Muhd-Anas/EchoID_anas.git
cd EchoID_anas

# Install required packages
pip install -r requirements.txt
```

### Dependencies Overview
```
keras==3.11.3          # Deep learning framework
librosa==0.11.0        # Audio processing
numpy==2.3.5           # Numerical computing
scikit_learn==1.4.2    # Machine learning utilities
sounddevice==0.5.3     # Audio I/O
soundfile==0.13.1      # Audio file I/O
vad==1.0.2             # Voice activity detection
ruamel.base==1.0.0     # YAML configuration
```

---

## 🚀 Usage

### 1. Data Preparation

Organize your audio data in the following structure:

```
data/
├── speaker0/          # Non-target speaker samples (negative class)
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── speaker1/          # Target speaker samples (positive class)
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

**Note**: If you have long recordings, proceed to step 2 for chunking.

---

### 2. Audio Chunking

Split long audio files into 3-second chunks for uniform training data:

```python
from src.data.audio_chunker import AudioChunker

# Initialize chunker
chunker = AudioChunker(chunk_duration_sec=3, sample_rate=16000)

# Process speaker0 recordings
chunker.run(
    speaker_dir="data/speaker0",
    output_dir="data/speaker0/chunks",
    file_count=21  # Number of files to process
)

# Process speaker1 recordings
chunker.run(
    speaker_dir="data/speaker1",
    output_dir="data/speaker1/chunks",
    file_count=21
)
```

**Important**: After chunking, move all generated `.wav` files from the `chunks/` subdirectories into their respective root speaker directories (`speaker0/`, `speaker1/`), then delete the original long recordings.

---

### 3. Training the Model

Run the complete training pipeline using `main.py`:

```python
python main.py
```

**What happens during training:**

1. **Load Dataset**: Audio files are loaded and split into train/test sets (80/20)
2. **Augment Training Data**: Waveform augmentation is applied (noise, pitch, amplitude)
3. **Convert to Mel-Spectrograms**: Audio is transformed into mel-spectrogram features
4. **Apply Mel Augmentation**: Additional augmentation on spectrograms
5. **Train CNN Model**: Model trains with early stopping and learning rate reduction
6. **Evaluate Performance**: Test set evaluation with comprehensive metrics

**Training Output:**
```
x_train_mel_aug shape: (40, 32, 64, 188, 1)
y_train_mel_aug shape: (40, 32)

Epoch 1/20
100/100 [==============================] - 45s 450ms/step
...
Accuracy: 0.95 | Precision: 0.94 | Recall: 0.96 | F1-Score: 0.95 | ROC-AUC: 0.98
```

**Trained models** are saved in: `models/cnn_model_v1/model_v1.keras`

---

### 4. Real-Time Inference

Launch the GUI application for live speaker recognition:

```python
python inference.py
```

**GUI Features:**
- 🎤 **Record Button**: Capture 3-second audio clips
- 📊 **Real-Time Prediction**: Instant speaker identification
- 🎯 **Confidence Score**: Display prediction probability
- 🔇 **VAD Integration**: Filter out silence and noise

**Example Output:**
```
✅ Target Speaker Detected! (Confidence: 92.3%)
❌ Non-Target Speaker (Confidence: 15.7%)
```

---

## 📂 Project Structure

```
EchoID_anas/
├── config.yaml                    # Global configuration file
├── main.py                        # Main training script
├── inference.py                   # Real-time inference launcher
├── requirements.txt               # Python dependencies
│
├── data/                          # Audio dataset directory
│   ├── speaker0/                  # Non-target speaker samples
│   └── speaker1/                  # Target speaker samples
│
├── models/                        # Trained model checkpoints
│   └── cnn_model_v1/             
│       └── model_v1.keras
│
├── notebooks/                     # Jupyter notebooks for experiments
│   ├── audio_preprocessing_experimental.ipynb
│   ├── model_training_experimental.ipynb
│   └── structure_experimental.ipynb
│
└── src/                           # Source code modules
    ├── data/                      # Data processing modules
    │   ├── audio_chunker.py       # Split long recordings
    │   ├── dataset_loader.py      # Load and batch audio
    │   ├── audio_augmentor.py     # Waveform augmentation
    │   └── mel_processor.py       # Mel-spectrogram conversion & augmentation
    │
    ├── models/                    # Model architecture & training
    │   ├── model_builder.py       # Dynamic CNN builder
    │   ├── trainer.py             # Training pipeline
    │   ├── callbacks.py           # Keras callbacks
    │   └── evaluation.py          # Model evaluation metrics
    │
    ├── inference/                 # Real-time inference
    │   └── listener.py            # GUI inference application
    │
    └── utils/                     # Utility functions
        ├── config_utils.py        # Config file parsing
        └── metrics_utils.py       # Custom metrics
```

---

## ⚙️ Configuration

The entire project is configurable via `config.yaml`. Key parameters:

### Model Architecture
```yaml
model:
  input_shape: [64, 188, 1]
  filters: [32, 64, 128]
  dense_units: [512]
  dropout_rates: [0.25, 0.25, 0.3, 0.5]
  kernel_sizes: [3, 3, 3]
  max_pool_sizes: [2, 2, 2]
```

### Training Parameters
```yaml
training:
  batch_size: 32
  epochs: 20
  validation_split: 0.2
  
  early_stopping:
    enable: true
    patience: 7
    monitor: val_loss
  
  reduce_lr:
    enable: true
    factor: 0.5
    patience: 4
    min_lr: 1e-6
```

### Inference Settings
```yaml
inference:
  duration: 3              # Chunk duration (seconds)
  sample_rate: 16000       # Audio sample rate (Hz)
  threshold: 0.7           # Confidence threshold
```

---

## 🤝 Contributing

Contributions are **welcome and encouraged**! Whether you're fixing bugs, improving documentation, or proposing new features, your input is valuable.

### How to Contribute

1. **Fork the repository**
2. **Create your feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Areas

We appreciate contributions in any of these areas:
- 🐛 Bug fixes and error handling improvements
- 📝 Documentation enhancements
- ✨ New feature implementations
- 🧪 Test coverage improvements
- 🎨 Code quality and refactoring
- 🌐 Multi-language support

**Have ideas but unsure how to implement them?** Open an issue to discuss! We're here to help.

---

## 🚧 Future Roadmap

### Planned Enhancements

#### 🎯 Core Features
- [ ] **Multi-Class Classification**: Extend from binary to multi-speaker recognition (3+ speakers)
- [ ] **Advanced Augmentation Methods**: 
  - [ ] Room impulse response (RIR) simulation
  - [ ] Background noise mixing
  - [ ] Speed perturbation
  - [ ] Codec simulation
- [ ] **Transfer Learning**: Leverage pre-trained models (VGGish, ResNet, wav2vec 2.0)

#### 🏗️ Architecture Improvements
- [ ] **Better Model Architecture**: 
  - [ ] Attention mechanisms
  - [ ] Residual connections
  - [ ] Deeper networks with batch normalization
- [ ] **Data Pipeline**: 
  - [ ] TensorFlow Dataset API integration
  - [ ] On-the-fly augmentation
  - [ ] Data caching for faster training

#### 📊 Model Evaluation
- [ ] **Cross-Validation**: K-fold validation for robust performance estimates
- [ ] **Confusion Matrix Visualization**: Detailed error analysis
- [ ] **Per-Speaker Performance**: Individual speaker accuracy metrics
- [ ] **Threshold Tuning**: ROC curve analysis for optimal thresholds

#### 🎤 Inference Enhancements
- [ ] **Web Interface**: Flask/FastAPI-based web app for browser-based inference
- [ ] **Batch Processing**: Process multiple audio files simultaneously
- [ ] **API Endpoints**: RESTful API for integration with other systems
- [ ] **Model Optimization**: Quantization and pruning for faster inference

#### 📦 Deployment
- [ ] **Docker Support**: Containerized deployment
- [ ] **Cloud Deployment**: AWS/GCP/Azure deployment guides
- [ ] **Mobile App**: TensorFlow Lite conversion for mobile inference
- [ ] **Edge Deployment**: Raspberry Pi / Jetson Nano support

#### 🛠️ Developer Experience
- [ ] **Automated Testing**: Unit and integration tests
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Pre-commit Hooks**: Code formatting and linting
- [ ] **Type Hints**: Full type annotation coverage
- [ ] **Better Logging**: Structured logging with log levels

### Community Goals
- [ ] **Comprehensive Tutorial Series**: Step-by-step video tutorials
- [ ] **Sample Datasets**: Public dataset links and benchmarks
- [ ] **Model Zoo**: Pre-trained models for different use cases
- [ ] **Paper/Blog Post**: Detailed technical write-up of methodology

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Muhd Uwais

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📬 Contact

**Muhd Uwais** (Muhd Anas) - Project Author

- 🌐 **Portfolio**: [https://nox-uwi.github.io/Form/](https://nox-uwi.github.io/Form/)
- 📧 **Email**: Contact via portfolio form
- 💼 **GitHub**: [@Muhd-Anas](https://github.com/Muhd-Anas)

### Feedback & Support

- **Found a bug?** Open an [issue](https://github.com/Muhd-Anas/EchoID_anas/issues)
- **Have a question?** Start a [discussion](https://github.com/Muhd-Anas/EchoID_anas/discussions)
- **Want to collaborate?** Reach out via the contact form on the portfolio site
- **Suggestions for improvement?** We'd love to hear them! Open a pull request or contact directly

---

## 🌟 Acknowledgments

Special thanks to:
- The **Keras** and **TensorFlow** teams for the deep learning framework
- **Librosa** developers for audio processing tools
- The open-source community for inspiration and support

---

## 🏆 Star this Project!

If **EchoID** helped you with your speaker recognition tasks or you found it useful for learning, please consider giving it a ⭐ on GitHub! Your support motivates continued development and helps others discover the project.

---

<p align="center">
  <strong>Built with ❤️ and 🧠 by an aspiring AI Developer</strong>
</p>

<p align="center">
  <b>#DPMG</b><br>
  <sub>Discipline • Peace • Myself • Growth</sub>
</p>

<p align="center">
  <sup>Happy Coding! 🚀</sup>
</p>
