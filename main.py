from src.data.dataset_loader import AudioDatasetLoader
from src.data.audio_augmentor import AudioAugmentor
from src.data.mel_processor import WaveformToMel, MelAugmentor
from models.trainer import Trainer
from src.models.evaluation import evaluate_model
from src.utils import save_config, bump_version



def main():
    # Load Dataset
    loader = AudioDatasetLoader(file_path="C:\\voice-speaker-binary-classifier\\data")
    x_train, x_test, y_train, y_test = loader.load_dataset()

    # Augment Data
    audio_augmentor = AudioAugmentor(sr=16000, batch_size=32)
    x_train_aug, y_train_aug = audio_augmentor.run(x_train, y_train, num_aug=3, shuffle=True)

    # Convert to Mel Spectrograms
    mel_processor = WaveformToMel()
    x_train_mel, y_train_mel = mel_processor.run(x_train_aug, y_train_aug)
    x_test_mel, y_test_mel = mel_processor.run(x_test, y_test)

    # Further Augmentation on Mel Spectrograms
    mel_augmentor = MelAugmentor()
    x_train_mel_aug, y_train_mel_aug = mel_augmentor.run(x_train_mel, y_train_mel, num_aug=2, shuffle=True)

    print(x_train_mel_aug.shape, y_train_mel_aug.shape)

    # Train Model
    trainer = Trainer()
    model, history = trainer.train(x_train_mel_aug, y_train_mel_aug, verbose=1, shuffle=True)

    # Evaluate Model
    metrics = evaluate_model(model, history, x_test_mel, y_test_mel)
    print(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['roc_auc'])


if __name__ == "__main__":
    print("Running Main Script...")
    main()    
    print("Main Script Finished.")