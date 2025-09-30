# train.py — Unsloth 4-bit QLoRA fine-tune with ~10 GiB VRAM target

import os, json, argparse, warnings, yaml
from typing import Dict, Any, List

# Import Unsloth FIRST to let it patch things
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

warnings.filterwarnings("ignore", category=UserWarning)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_dataset(train_file: str, eval_file: str) -> Dict[str, Dataset]:
    train_rows = read_jsonl(train_file)
    eval_rows = read_jsonl(eval_file) if os.path.exists(eval_file) else []
    return {
        "train": Dataset.from_list(train_rows),
        "eval": Dataset.from_list(eval_rows) if eval_rows else None,
    }

def format_chat(tokenizer, example):
    user, assistant = None, None
    for m in example.get("messages", []):
        role = m.get("role"); content = (m.get("content") or "").strip()
        if role == "user" and user is None:
            user = content
        elif role == "assistant" and assistant is None:
            assistant = content
    if not user or not assistant:
        return {"text": None}
    eos = tokenizer.eos_token or "</s>"
    return {"text": f"User: {user}\nAssistant: {assistant}{eos}"}

class CompletionOnlyCollator:
    """Mask everything before 'Assistant:' to -100 so loss only covers the answer."""
    def __init__(self, tokenizer, response_template="Assistant:"):
        self.tok = tokenizer
        self.response_template = response_template

    def __call__(self, batch):
        texts = [ex["text"] for ex in batch]
        enc = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.tok.model_max_length, return_tensors="pt"
        )
        labels = enc["input_ids"].clone()
        for i, text in enumerate(texts):
            j = text.find(self.response_template)
            if j == -1:
                labels[i, :] = -100
                continue
            prefix_ids = self.tok(text[:j], add_special_tokens=False).input_ids
            labels[i, :len(prefix_ids)] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    max_seq_len = int(cfg["max_seq_len"])
    target_vram_gib = float(cfg.get("target_vram_gib", 10))

    # Optional: a gentle allocator hint that often smooths fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Build a max_memory map to nudge HF’s device_map sharding to ~10 GiB
    # Works when device_map="auto"; Unsloth forwards kwargs to HF under the hood.
    max_memory = {0: f"{int(target_vram_gib)}GiB"}  # single-GPU run

    # Load via Unsloth; let it return its tokenizer (simplest/most compatible)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        max_memory=max_memory,  # soft cap
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply QLoRA adapters (no PEFT/TRL imports required)
    l = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        use_rslora=l.get("use_rslora", True),
        use_gradient_checkpointing=l.get("gradient_checkpointing", "unsloth"),
        bias="none",
    )

    # Data
    files = cfg["train"]
    ds = build_dataset(files["train_file"], files["eval_file"])
    train_ds = ds["train"].map(lambda ex: format_chat(tokenizer, ex)).filter(lambda ex: ex["text"] is not None)
    eval_ds = None
    if ds["eval"] is not None:
        eval_ds = ds["eval"].map(lambda ex: format_chat(tokenizer, ex)).filter(lambda ex: ex["text"] is not None)

    collator = CompletionOnlyCollator(tokenizer)

    t = cfg["train"]
    output_dir = t["output_dir"]; os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        bf16=t.get("bf16", True),
        fp16=False,
        optim="paged_adamw_8bit",   # 4-bit friendly
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        report_to="none",
        seed=t["seed"],
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=t["eval_steps"] if eval_ds is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    ckpt_dir = os.path.join(output_dir, "checkpoint")
    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(output_dir)
    print("\nTraining complete.")
    print("Adapter saved to:", ckpt_dir)
    print("Tokenizer saved to:", output_dir)

if __name__ == "__main__":
    main()
