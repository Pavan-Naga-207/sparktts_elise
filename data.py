# elise_to_jsonl_legacy_bytes.py
import os, json, tempfile, pathlib
from tqdm import tqdm
import pyarrow as pa
import datasets
import numpy as np
import torch
import soundfile as sf  # needs libsndfile
from sparktts.models.audio_tokenizer import BiCodecTokenizer

START = {
    "task": "<|task_tts|>",
    "content_s": "<|start_content|>",
    "content_e": "<|end_content|>",
    "global_s": "<|start_global_token|>",
    "global_e": "<|end_global_token|>",
    "sem_s": "<|start_semantic_token|>",
    "sem_e": "<|end_semantic_token|>", 
    "end": "<|im_end|>",
}

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def ids_to_tag_string(prefix, ids):
    if torch.is_tensor(ids): ids = ids.reshape(-1).tolist()
    else: ids = list(ids)
    return "".join(f"<|{prefix}_{int(i)}|>" for i in ids)

def build_prompt(text, g_ids, s_ids):
    return "".join([
        START["task"], START["content_s"], text, START["content_e"],
        START["global_s"], ids_to_tag_string("bicodec_global", g_ids), START["global_e"],
        START["sem_s"], ids_to_tag_string("bicodec_semantic", s_ids), START["sem_e"],
        START["end"],
    ])

def main():
    out_dir = "output_prompt"
    out_jsonl = os.path.join(out_dir, "Elise.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device()
    print("Using device:", device)
    tok = BiCodecTokenizer("pretrained_models/Spark-TTS-0.5B", device=device)

    ds = datasets.load_dataset("MrDragonFox/Elise", split="train")
    tbl: pa.Table = ds.data
    audio_col = tbl.column("audio")
    text_col = tbl.column("text") if "text" in tbl.column_names else None

    fields = [f.name for f in audio_col.type]
    if "bytes" not in fields:
        raise SystemExit(
            "This cached Elise build has no audio.bytes. Either re-download with a newer datasets "
            "or use the earlier 'upgrade + soundfile' path."
        )

    tmp_root = tempfile.mkdtemp(prefix="elise_bytes_")
    print("Temp dir:", tmp_root)

    n_ok, n_err = 0, 0
    with open(out_jsonl, "w", encoding="utf-8") as outf:
        for i in tqdm(range(len(tbl)), desc="Processing Elise"):
            try:
                s = audio_col[i]         # StructScalar
                a = s.as_py()            # {'bytes': ..., 'path': 'eaa31ab9.wav'}
                raw = a.get("bytes", None)
                pth = a.get("path", "")
                if not raw:
                    n_err += 1
                    continue

                # choose an extension from the path if present
                ext = pathlib.Path(pth).suffix.lower() if pth else ".wav"
                if ext not in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
                    ext = ".wav"

                # write the original bytes first
                orig_path = os.path.join(tmp_root, f"{i:06d}{ext}")
                with open(orig_path, "wb") as w:
                    w.write(raw)

                # ensure WAV for tokenizer
                wav_path = orig_path
                if ext != ".wav":
                    data, sr = sf.read(orig_path, dtype="float32")
                    wav_path = os.path.join(tmp_root, f"{i:06d}.wav")
                    sf.write(wav_path, data, sr)

                # tokenize and write JSONL line
                g_ids, s_ids = tok.tokenize(wav_path)
                text = (text_col[i].as_py() if text_col is not None else "") or ""
                text = text.strip()

                outf.write(json.dumps({"text": build_prompt(text, g_ids, s_ids)}, ensure_ascii=False) + "\n")
                n_ok += 1

                # cleanup per sample to save disk
                try:
                    if wav_path != orig_path and os.path.exists(wav_path): os.remove(wav_path)
                    if os.path.exists(orig_path): os.remove(orig_path)
                except Exception:
                    pass

            except Exception as e:
                n_err += 1
                print(f"[ERR {i}] {e}")

    print(f"Done. Wrote {n_ok} lines to {out_jsonl}. Errors: {n_err}")

if __name__ == "__main__":
    main()