"""
Fine-tune a Llama 3.x Instruct model locally using the same prompt+data
format as the OpenAI fine-tuning pipeline (train/val JSONL with `messages`).

Expected dataset record format (per line in JSONL):
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"corrected\": \"...\"}"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import settings
from lib.utils import setup_log

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _messages_to_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Convert OpenAI-style `messages` to a single training string using the model's chat template.
    Falls back to a simple concatenation if the tokenizer doesn't provide a chat template.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Fallback: minimal, deterministic formatting.
    chunks: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chunks.append(f"[{role}]\n{content}\n")
    return "\n".join(chunks).strip() + "\n"


@dataclass(frozen=True)
class TrainArgs:
    model_id: str
    train_file: str
    val_file: str
    output_dir: str
    max_seq_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    seed: int
    use_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    merge_lora: bool


def _parse_args() -> TrainArgs:
    default_output_dir = os.path.join("data", "output", "job", settings.run_id, "llama")

    p = argparse.ArgumentParser(description="Fine-tune Llama locally (LoRA/QLoRA) on GEC chat data.")
    p.add_argument(
        "--model_id",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HF model id or local path (e.g. meta-llama/Llama-3.3-70B-Instruct).",
    )
    p.add_argument("--train_file", default=settings.dataset_train_filename, help="Training JSONL (messages format).")
    p.add_argument("--val_file", default=settings.dataset_val_filename, help="Validation JSONL (messages format).")
    p.add_argument("--output_dir", default=default_output_dir, help="Output directory for adapter/model.")

    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--use_4bit",
        action="store_true",
        help="Enable 4-bit QLoRA loading via bitsandbytes (recommended for large models).",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base weights at the end.")

    ns = p.parse_args()
    return TrainArgs(**vars(ns))


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM

        from peft import LoraConfig, PeftModel, get_peft_model
        from trl import SFTTrainer, SFTConfig
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing local fine-tuning dependencies. Install at least:\n"
            "  pip install -U transformers datasets accelerate peft trl\n"
            "And for QLoRA:\n"
            "  pip install -U bitsandbytes\n"
            f"\nOriginal import error: {e}"
        )

    logger.info("Loading datasets...")
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not os.path.exists(args.val_file):
        raise FileNotFoundError(f"Val file not found: {args.val_file}")

    train_raw = _read_jsonl(args.train_file)
    val_raw = _read_jsonl(args.val_file)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text_records(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for r in rows:
            messages = r.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            out.append({"text": _messages_to_text(tokenizer, messages)})
        return out

    train_ds = Dataset.from_list(to_text_records(train_raw))
    eval_ds = Dataset.from_list(to_text_records(val_raw))

    logger.info("Loading model...")
    quantization_config = None
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        except Exception as e:
            raise SystemExit(f"--use_4bit requested but bitsandbytes/BitsAndBytesConfig unavailable: {e}")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    logger.info("Attaching LoRA adapters...")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # Works for Llama-family models with standard naming
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving adapter to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_lora:
        logger.info("Merging LoRA into base model...")
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        base = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto" if torch.cuda.is_available() else None)
        peft_model = PeftModel.from_pretrained(base, args.output_dir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info("Merged model saved to %s", merged_dir)

    logger.info("Done.")


if __name__ == "__main__":
    setup_log()
    main()
