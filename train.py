#!/usr/bin/env python3

import os
import math
import argparse
import yaml
import torch
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)



def _extract_texts(seq):
    out = []
    for x in seq:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for k in ("text", "prompt", "content"):
                v = x.get(k)
                if isinstance(v, str):
                    out.append(v)
                    break
    return out


def load_yaml_prompts(path: str):
    """
    Now supports: .yaml/.yml, .json, .jsonl
    Returns list[str] of prompts.
    """
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        texts = []
        with open(p, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, (list, tuple)):
                    texts.extend(_extract_texts(obj))
                elif isinstance(obj, dict):
                    texts.extend(_extract_texts([obj]))
                elif isinstance(obj, str):
                    texts.append(obj)
        return texts

    if suffix == ".json":
        with open(p, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return _extract_texts(obj)
        if isinstance(obj, dict):
            for key in ("data", "items", "samples", "records"):
                if key in obj and isinstance(obj[key], list):
                    return _extract_texts(obj[key])
            return _extract_texts([obj])
        raise ValueError("Unsupported JSON structure.")

    with open(p, "r") as f:
        content = yaml.safe_load(f)
    if isinstance(content, list):
        return _extract_texts(content)
    if isinstance(content, dict):
        for key in ("data", "items", "samples", "records"):
            if key in content and isinstance(content[key], list):
                return _extract_texts(content[key])
        return _extract_texts([content])
    raise ValueError("Unsupported YAML structure.")


def get_device_and_dtype():
    """
    Force CUDA if available; fallback to CPU.
    Use bfloat16 when on CUDA (standard for modern full finetuning).
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return "cuda", torch.float32
    else:
        return "cpu", torch.float32


def load_yaml_prompts_from_list(lst: List[Any]) -> List[str]:
    out = []
    for x in lst:
        if isinstance(x, dict) and "text" in x and isinstance(x["text"], str):
            out.append(x["text"])
        elif isinstance(x, str):
            out.append(x)
    return out



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_yaml", type=str, required=True,
                        help="Path to your processed prompts YAML file (the one you just created).")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="HF model id or local path. Example: Spark-TTS-0.5B")
    parser.add_argument("--output_dir", type=str, default="./sparktts-full-finetuned")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Truncate/pack to this length. Make sure it fits your model.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device train batch size. Keep tiny on Mac.")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size*grad_accum).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for full finetuning.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    FORCE_FULL_FINETUNE = True

    device, torch_dtype = get_device_and_dtype()
    print(f"Using device={device}, dtype={torch_dtype}")

    texts = load_yaml_prompts(args.prompts_yaml)
    if not texts:
        raise RuntimeError("No prompts found in YAML.")
    print(f"Loaded {len(texts)} prompts")

    dataset = Dataset.from_dict({"text": texts})
    train_test = dataset.train_test_split(test_size=max(1, int(0.05 * len(dataset))), seed=args.seed)
    train_ds, eval_ds = train_test["train"], train_test["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    emotion_tokens = [f"<{e}>" for e in [
    "giggles",
    "laughs",
    "long pause",
    "chuckles",
    "whispers",
    "normal volume",
    "sighs",
    "clicks tongue",
    "gasps",
    "moans",
    "sonora",
    "habla en ingles",
    "smacks lips",
    "sigh",
    "chewing",
    "sensual music",
    "clears throat",
    "singing",
    "stutters",
    "breathes deeply",
    "laughs nervously",
    "kissing noise",
    "burps",
    "sadly",
    "sniffs",
    "scoffs",
    "music",
    "finger cracks",
    "smooches",
    "yawning",
    "mouth sounds",
    "exhales",
    "fabric rustling",
    "trails off",
    "nervous laughter",
    "mouchoir eternue",
    "bruit_de bouche",
    "hugging sound",
    "romantic music playing",
    "coughs",
    "tapping sounds",
]]
    tokenizer.add_special_tokens({'additional_special_tokens': emotion_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if FORCE_FULL_FINETUNE:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full finetune: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M params trainable")

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        optim="adamw_torch_fused",
        gradient_checkpointing=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting full finetuning...")
    train_result = trainer.train()

    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    try:
        metrics["eval_ppl"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        metrics["eval_ppl"] = float("inf")

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved full model to: {args.output_dir}")


if __name__ == "__main__":
    main()