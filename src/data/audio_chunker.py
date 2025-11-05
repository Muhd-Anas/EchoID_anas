# Voice Speaker Recognition - Audio Splitting Module
"""
This module handles splitting long audio recordings into smaller, fixed-duration chunks
to prepare data for speaker recognition model training.

Note:
After the chunking process is complete, move all generated WAV files from the chunk output
folders into their respective root speaker directories (e.g., `speaker0` and `speaker1`).
Then delete the original non-chunked audio files to maintain a clean and consistent
dataset structure for later preprocessing and training steps.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Audio Chunk Creation
License: MIT
"""

import os
import logging
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


# ------------------ Module Logger ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AudioChunker:
    """
    AudioChunker
    ------------
    A class to split long audio recordings into smaller fixed-length chunks
    for dataset preparation in speaker recognition or speech classification tasks.

    It processes all `.wav` or `.m4a` files in a given speaker directory,
    generates uniform 3-second (or configurable) segments, and saves them
    into an output folder.

    Typical Workflow
    ----------------
    1. Initialize the class with desired parameters (sample rate, chunk duration, etc.)
    2. Call the `run()` method with input/output directories and file count.
    3. The method will process all files, generate chunks, and log progress.

    Attributes
    ----------
    sample_rate : int
        Target sampling rate for all processed audio files
    chunk_duration_sec : int
        Duration of each output chunk in seconds

    Example
    -------
    >>> from audio_chunker import AudioChunker
    >>>
    >>> chunker = AudioChunker(sample_rate=16000, chunk_duration_sec=3)
    >>> chunker.run(
    ...     speaker_dir="C:/voice-speaker-binary-classifier/data/speaker0",
    ...     output_dir="C:/voice-speaker-binary-classifier/data/speaker0/chunks",
    ...     file_count=21
    ... )
    >>>
    >>> print("Chunking completed successfully!")

    Note
    ----
    After chunking, move all generated WAV files into their root speaker directories
    (e.g., `speaker0`, `speaker1`) before running further preprocessing modules.
    You can delete the original unchunked files to maintain a clean dataset structure.
    """

    def __init__(self, chunk_duration_sec: int = 3, sample_rate: int = 16000):
        """
        Initialize the AudioChunker.

        Args:
            chunk_duration_sec (int): Duration of each chunk in seconds (default: 3)
            sample_rate (int): Target sample rate for audio (default: 16000)
        """

        self.chunk_duration_sec = chunk_duration_sec
        self.sample_rate = sample_rate
        logger.info("Initialized AudioChunker with chunk size %d sec, sample rate %d Hz",
                    chunk_duration_sec, sample_rate)

    def _create_audio_chunks(
        self,
        input_file: str,
        output_dir: str,
        counter_start: int = 0,
    ) -> int:
        """
        Split an audio file into fixed-duration chunks for training data preparation.

        Args:
            input_file (str): Path to the input audio file.
            output_dir (str): Directory to save the generated chunks.
            counter_start (int): Starting counter for chunk naming (default: 0).

        Returns:
            int: Updated counter after processing all chunks.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            Exception: For other audio processing errors.
        """
        try:
            # --- Validate input ---
            if not Path(input_file).exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # --- Load audio ---
            logger.info("📁 Loading audio file: %s", input_file)
            audio_data, sr = librosa.load(
                input_file, sr=self.sample_rate, mono=True)

            # --- Chunk parameters ---
            samples_per_chunk = int(self.chunk_duration_sec * sr)
            total_chunks = len(audio_data) // samples_per_chunk
            logger.info("Audio length: %.2fs | Chunks to create: %d",
                        len(audio_data) / sr, total_chunks)

            # --- Ensure output directory exists ---
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            counter = counter_start
            chunks_created = 0

            # --- Split and save chunks ---
            for i in range(total_chunks):
                start_sample = i * samples_per_chunk
                end_sample = start_sample + samples_per_chunk
                chunk = audio_data[start_sample:end_sample]

                chunk_filename = f"voice_{counter:03d}.wav"
                output_path = Path(output_dir) / chunk_filename
                sf.write(output_path, chunk, sr)

                counter += 1
                chunks_created += 1

            logger.info(
                f"✅ Successfully created {chunks_created} chunks from {input_file}")
            return counter

        except FileNotFoundError:
            logger.error(f"❌ File not found: {input_file}")
            raise
        except Exception as e:
            logger.error(f"❌ Error processing {input_file: {str(e)}}")
            raise

# ---------------------------------------------------------

    def _process_all_speaker_files(
        self,
        speaker_dir: str,
        output_dir: str,
        file_count: int,
        counter_start: int = 0
    ):
        """
        Process all audio files in a speaker directory (e.g., Voice (1).m4a ... Voice (n).m4a).

        Args:
            speaker_dir (str): Directory containing original audio files.
            output_dir (str): Directory to save processed chunks.
            file_count (int): Number of files to process.
            counter_start (int): Starting number for chunk naming.

        Returns:
            int: Total number of chunks created for the speaker.
        """

        logger.info("🎤 Processing speaker directory: %s", speaker_dir)
        logger.info("=" * 50)

        counter = counter_start
        processed_files = 0

        # Process files Voice (1).m4a, Voice (2).m4a, etc.
        for i in range(1, file_count + 1):
            file_path = os.path.join(speaker_dir, f"Voice ({i}).m4a")
            print(file_path)

            # Check if file exists before trying to process it
            if Path(file_path).exists():
                counter = self._create_audio_chunks(
                    input_file=file_path,
                    output_dir=output_dir,
                    counter_start=counter
                )
                processed_files += 1
            else:
                logger.warning(
                    "⚠️ File not found: Voice (%d).m4a, skipping...", i)

        logger.info(
            f"\n📊 Summary for {speaker_dir}: • Files processed: {processed_files} • Total chunks created: {counter}")
        return counter

    # ---------------------------------------------------------

    def run(
            self,
            speaker_dir: str,
            output_dir: str,
            file_count: int,
    ) -> bool:
        """
        Run the complete chunking pipeline for a single speaker.

        Args:
            speaker_dir (str): Directory containing the original audio files.
            output_dir (str): Directory to save processed chunks.
            file_count (int): Number of files to process in the speaker folder.

        Returns:
            bool: True if the process completes successfully.
        """

        logger.info("🚀 Starting Audio Chunking Pipeline")
        logger.info("Splitting audio files in '%s' into %d-second chunks",
                    speaker_dir, self.chunk_duration_sec)
        logger.info("=" * 50)

        # Process the speaker's directory
        self._process_all_speaker_files(
            speaker_dir=speaker_dir,
            output_dir=output_dir,
            file_count=file_count,
        )

        print(f"\n{'=' * 50}")
        logger.info("🎉 Audio preprocessing completed!")
        print("\n✨ Your chunks are ready for further preprocessing steps.")
        print("\nNext steps:")
        print("➡️ Move all chunks into their root directories (e.g., speaker0, speaker1) before proceeding.")
        return True


# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
