"""Запуск QLoRA-адаптера Qwen; требует NVIDIA CUDA и bitsandbytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings


def main() -> None:
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

    root = settings.MODEL_DIR / "qwen_qlora_experiment_v1"
    config = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(settings.QWEN_LOCAL_PATH, local_files_only=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        settings.QWEN_LOCAL_PATH, local_files_only=True, quantization_config=quantization, device_map="auto",
    )
    dataset = load_dataset("json", data_files={"train": str(root / "train.jsonl"), "validation": str(root / "validation.jsonl")})

    def format_row(row: dict) -> str:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)

    hp = config["hyperparameters"]
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        formatting_func=format_row,
        peft_config=LoraConfig(r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=hp["dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
        args=TrainingArguments(output_dir=str(root / "adapter"), num_train_epochs=hp["epochs"], learning_rate=hp["learning_rate"], per_device_train_batch_size=1, gradient_accumulation_steps=16, eval_strategy="epoch", save_strategy="epoch", logging_steps=10, bf16=True, report_to="none"),
    )
    trainer.train()
    trainer.save_model(str(root / "adapter"))
    print("all_done_train_qwen_qlora.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_train_qwen_qlora.py")
        raise
