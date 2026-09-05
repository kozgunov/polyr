"""Запуск QLoRA-адаптера Qwen; требует NVIDIA CUDA и bitsandbytes."""

from __future__ import annotations

import json
import sys
import argparse
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--task", choices=("direction", "entry", "exit"), default="direction")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLORA_BLOCKED_NO_CUDA: текущий torch CPU-only. Используйте NVIDIA CUDA/Colab; "
            "подготовленные train.jsonl и validation.jsonl уже готовы."
        )
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    if args.bundle:
        root = args.bundle
        config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
        hp = {"rank": 16, "alpha": 32, "dropout": 0.05, "epochs": 2, "learning_rate": 0.0001, "max_length": 768}
        train_path = root / f"qwen_{args.task}_train.jsonl"
        validation_path = root / f"qwen_{args.task}_validation.jsonl"
        adapter_path = root / "candidates" / f"qwen_{args.task}_lora_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    else:
        root = settings.MODEL_DIR / "qwen_qlora_experiment_v1"
        config = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        hp = config["hyperparameters"]
        train_path, validation_path = root / "train.jsonl", root / "validation.jsonl"
        adapter_path = root / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(settings.QWEN_LOCAL_PATH, local_files_only=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        settings.QWEN_LOCAL_PATH, local_files_only=True, quantization_config=quantization, device_map="auto",
    )
    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(validation_path)})

    def format_row(row: dict) -> str:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        formatting_func=format_row,
        peft_config=LoraConfig(r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=hp["dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
        args=TrainingArguments(output_dir=str(adapter_path), num_train_epochs=hp["epochs"], learning_rate=hp["learning_rate"], per_device_train_batch_size=1, gradient_accumulation_steps=16, eval_strategy="epoch", save_strategy="epoch", logging_steps=10, bf16=True, report_to="none"),
    )
    trainer.train()
    trainer.save_model(str(adapter_path))
    print("all_done_train_qwen_qlora.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_train_qwen_qlora.py")
        raise
