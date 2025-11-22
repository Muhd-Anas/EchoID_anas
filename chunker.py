# Voice Speaker Recognition - Audio Chunker CLI
"""
This module serves as the command-line entry point for the audio chunking pipeline.
It utilizes the `AudioChunker` class to process raw audio recordings into 
uniform 3-second segments suitable for training.

Usage:
------
    python chunker.py <files_for_speaker0> <files_for_speaker1>

    Example: python chunker.py 21 21
    (This will process 21 audio files for both speaker0 and speaker1)

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Chunking Execution (CLI Entry Point)
License: MIT
"""

import argparse
from src.data.audio_chunker import AudioChunker


# =============================================================
# Main Execution Function
# =============================================================

def main():
    """
    Main entry point for the audio chunking script.

    Parses command-line arguments to determine how many files to process 
    for each speaker and triggers the chunking pipeline.
    """

    # ------------------ Argument Parsing ------------------
    parser = argparse.ArgumentParser(
        description="Split long audio recordings into uniform 3-second chunks."
    )

    parser.add_argument(
        "number0",
        type=int,
        help="Number of files to process for Speaker0"
    )
    parser.add_argument(
        "number1",
        type=int,
        help="Number of files to process for Speaker1"
    )

    args = parser.parse_args()

    # ------------------ Pipeline Initialization ------------------
    # Initialize chunker with standard project settings (3s @ 16kHz)
    chunker = AudioChunker(chunk_duration_sec=3, sample_rate=16000)

    # ------------------ Processing Speaker 0 ------------------
    # Process negative class samples
    chunker.run(
        speaker_dir="data/speaker0",
        output_dir="data/speaker0/chunks",
        file_count=args.number0
    )

    # ------------------ Processing Speaker 1 ------------------
    # Process target class samples
    chunker.run(
        speaker_dir="data/speaker1",
        output_dir="data/speaker1/chunks",
        file_count=args.number1
    )


# ---------------------------------------------------------
# Script Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🔄 Running Audio Chunker...")
    main()
    print("✅ Audio Chunker Finished.")
