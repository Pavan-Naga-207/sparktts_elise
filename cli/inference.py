# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform

from cli.SparkTTS import SparkTTS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--text", type=str, required=True, help="Text for TTS generation"
    )
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    return parser.parse_args()

def run_test(args):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    texts = [
            "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
            "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
            "produced the block books, which were the immediate predecessors of the true printed book,",
            "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
            "And it is worth mention in passing that, as an example of fine typography",
            "the earliest book printed with movable types, the Gutenberg, or forty-two line Bible of about 1455,",
            "Printing, then, for our purpose, may be considered as the art of making books by means of movable types",
            "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress,",
            "it is of the first importance that the letter used should be fine in form;",
            "especially as no more time is occupied, or cost incurred, in casting, setting, or printing beautiful letters"
        ]
    if torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    # Initialize the model
    model = SparkTTS(args.model_dir, device)
    for idx, text in enumerate(texts):
        save_path = os.path.join(args.save_dir, f"test_{idx+1}.wav")
        logging.info(f"Generating audio for text {idx+1}/{len(texts)}")
        with torch.no_grad():
            wav = model.inference(
                text,
                args.prompt_speech_path,
                prompt_text=args.prompt_text,
                gender=args.gender,
                pitch=args.pitch,
                speed=args.speed,
            )
            sf.write(save_path, wav, samplerate=16000)
          
        

def run_tts(args):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    # if platform.system() == "Darwin" and torch.backends.mps.is_available():
    #     # macOS with MPS support (Apple Silicon)
    #     device = torch.device(f"mps:{args.device}")
    #     logging.info(f"Using MPS device: {device}")
    if torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    # Initialize the model
    model = SparkTTS(args.model_dir, device)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            args.text,
            args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    run_tts(args)
