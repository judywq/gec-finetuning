"""
Inference with a locally fine-tuned Llama model produced by `E01_llama_finetune.py`.

Reads the same test JSONL format as the rest of the repo:
{
  "messages": [{"role":"user","content":"..."}],
  "metadata": {...}
}

Writes one JSON array per line to match the parallel inference scripts:
[
  {"messages": [...]},
  {"choices": [{"message": {"content": "..."}}], "model": "..."},
  {...metadata...}
]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import settings
from lib.utils import backup_output_file, setup_log

logger = logging.getLogger(__name__)


def _read_jsonl_lines(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _write_jsonl_results(path: str, rows: List[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Args:
    input: str
    output: str
    base_model_id: str
    adapter_dir: Optional[str]
    model_dir: Optional[str]
    use_4bit: bool
    temperature: float
    max_new_tokens: int
    top_p: float
    do_sample: bool
    seed: int


def _parse_args() -> Args:
    default_adapter_dir = os.path.join("data", "output", "job", settings.run_id, "llama")
    p = argparse.ArgumentParser(description="Run local Llama inference on test JSONL.")
    p.add_argument("--input", type=str, default=settings.dataset_test_filename, help="Input JSONL file path")
    p.add_argument(
        "--output",
        type=str,
        default="data/output/result/test_result_llama_finetuned_local.jsonl",
        help="Output JSONL file path",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--adapter_dir",
        type=str,
        default=default_adapter_dir,
        help="LoRA adapter directory produced by E01 (default: data/output/job/<run_id>/llama).",
    )
    group.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Merged/full model directory (e.g. <adapter_dir>/merged). If set, adapter_dir is ignored.",
    )

    p.add_argument(
        "--base_model_id",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HF base model id/path (needed when using --adapter_dir).",
    )
    p.add_argument("--use_4bit", action="store_true", help="Load base model in 4-bit (requires bitsandbytes).")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (otherwise greedy).")
    p.add_argument("--seed", type=int, default=42)

    ns = p.parse_args()
    return Args(**vars(ns))


def main() -> None:
    args = _parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing inference dependencies. Install at least:\n"
            "  pip install -U transformers accelerate\n"
            "And for adapters:\n"
            "  pip install -U peft\n"
            "And for 4-bit:\n"
            "  pip install -U bitsandbytes\n"
            f"\nOriginal import error: {e}"
        )

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    prev = backup_output_file(args.output)
    logger.info("Backup file: %s --> %s", args.output, prev)

    rows = _read_jsonl_lines(args.input)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Loading tokenizer...")
    model_id_for_tokenizer = args.model_dir or args.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.model_dir:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
        loaded_model_name = args.model_dir
    else:
        if not args.adapter_dir:
            raise ValueError("Either --model_dir or --adapter_dir must be provided.")
        if not os.path.exists(args.adapter_dir):
            raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")

        if args.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except Exception as e:
                raise SystemExit(f"--use_4bit requested but bitsandbytes/BitsAndBytesConfig unavailable: {e}")

        base = AutoModelForCausalLM.from_pretrained(args.base_model_id, **model_kwargs)
        try:
            from peft import PeftModel
        except Exception as e:
            raise SystemExit(f"Adapter loading requires peft. Import error: {e}")
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        loaded_model_name = f"{args.base_model_id} + {args.adapter_dir}"

    model.eval()

    results: List[Any] = []
    for item in rows:
        messages = item.get("messages", [])
        metadata = item.get("metadata", {}) or {}
        if not isinstance(messages, list) or len(messages) == 0:
            results.append([{"messages": []}, {"error": "Missing messages"}, metadata])
            continue

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple prompt when chat template is unavailable
            prompt = "\n".join(m.get("content", "") for m in messages) + "\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        new_tokens = gen[0, input_ids.shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        response = {
            "id": None,
            "object": "chat.completion",
            "model": loaded_model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }
        results.append([{"messages": messages}, response, metadata])

    _write_jsonl_results(args.output, results)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    setup_log()
    main()

