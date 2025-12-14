import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import TASK_TOKEN_MAP

model_path = "spark_elise"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
audio_tokenizer = BiCodecTokenizer(model_path)

special_tokens = [
    "[angry]",
    "[curious]",
    "[excited]",
    "[giggle]",
    "[laughs harder]",
    "[laughs]",
    "[screams]",
    "[sighs]",
    "[sings]",
    "[whispers]",
]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

try:
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
except KeyError:
    print("Warning: '<|im_end|>' not in tokenizer vocab. Using default EOS token.")
    eos_token_id = tokenizer.eos_token_id

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = eos_token_id


def _ensure_audio_tokenizer_device(device: torch.device) -> None:
    """Move tokenizer components to the requested device in-place."""
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    if hasattr(audio_tokenizer, "feature_extractor"):
        audio_tokenizer.feature_extractor.to(device)


def _serialize_audio_tokens(token_ids: torch.Tensor, token_type: str) -> str:
    flat_ids = token_ids.detach().cpu().reshape(-1).tolist()
    return "".join([f"<|bicodec_{token_type}_{int(idx)}|>" for idx in flat_ids])


def _build_basic_tts_prompt(text: str) -> str:
    """Build prompt for basic TTS (model generates voice characteristics)."""
    return "".join([
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>",
    ])


def _build_cloning_prompt(
    text: str,
    prompt_speech_path: Path,
    prompt_text: Optional[str],
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    """Build prompt for voice cloning from reference audio."""
    if not prompt_speech_path.exists():
        raise FileNotFoundError(f"Prompt audio not found: {prompt_speech_path}")

    _ensure_audio_tokenizer_device(device)
    global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(str(prompt_speech_path))
    global_tokens = _serialize_audio_tokens(global_token_ids, "global")

    inputs = [
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
    ]
    if prompt_text:
        inputs.append(prompt_text)
    inputs.extend([
        text,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
    ])
    if prompt_text:
        semantic_tokens = _serialize_audio_tokens(semantic_token_ids, "semantic")
        inputs.extend(["<|start_semantic_token|>", semantic_tokens])

    return "".join(inputs), global_token_ids


def _format_global_tokens(global_tokens: torch.Tensor) -> torch.Tensor:
    if global_tokens.dim() == 1:
        return global_tokens.unsqueeze(0)
    if global_tokens.dim() > 2:
        return global_tokens.squeeze(0)
    return global_tokens


@torch.inference_mode()
def generate_speech_from_text(
    text: str,
    prompt_speech_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1,
    max_new_audio_tokens: int = 2048,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    """Generate speech from text. Uses voice cloning if reference audio provided, else basic TTS."""
    torch.compiler.reset()

    use_cloning = prompt_speech_path is not None

    if use_cloning:
        prompt_audio_path = Path(prompt_speech_path)
        prompt, prompt_global_token_ids = _build_cloning_prompt(
            text, prompt_audio_path, prompt_text, device
        )
    else:
        prompt = _build_basic_tts_prompt(text)
        prompt_global_token_ids = None

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)

    print("Generating token sequence...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_audio_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Token sequence generated.")

    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        print("Warning: No semantic tokens found in the generated output.")
        return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    
    if use_cloning:
        global_token_tensor = _format_global_tokens(prompt_global_token_ids.to(device))
    else:
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        if not global_matches:
            print("Warning: No global tokens found in the generated output.")
            return np.array([], dtype=np.float32)
        global_token_tensor = torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0)

    print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
    print(f"Found {global_token_tensor.shape[-1]} global tokens.")

    print("Detokenizing audio tokens...")
    _ensure_audio_tokenizer_device(device)
    wav_np = audio_tokenizer.detokenize(
        _format_global_tokens(global_token_tensor.to(device)),
        pred_semantic_ids.to(device),
    )
    print("Detokenization complete.")

    return wav_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech from text (basic TTS or voice cloning).")
    parser.add_argument(
        "--text",
        type=str,
        default="Sometimes when I'm out in nature, I feel this incredible sense of peace.",
        help="The text to be converted to speech.",
    )
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to reference audio for voice cloning. If omitted, uses basic TTS.",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        help="Transcript of the reference audio. Improves cloning similarity when provided.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="generated_speech.wav",
        help="The path to save the generated audio file.",
    )
    args = parser.parse_args()

    mode = "voice cloning" if args.prompt_speech_path else "basic TTS"
    print(f"Generating speech for: '{args.text}' using {mode}")
    if args.prompt_speech_path:
        print(f"Reference audio: {args.prompt_speech_path}")

    generated_waveform = generate_speech_from_text(
        args.text,
        prompt_speech_path=args.prompt_speech_path,
        prompt_text=args.prompt_text,
    )

    if generated_waveform.size > 0:
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(args.output_path, generated_waveform, sample_rate)
        print(f"Audio saved to {args.output_path}")
    else:
        print("Audio generation failed (no tokens found?).")