"""Cumulative LoRA SFT of the defender on one round's harvested (jailbreak
prompt -> safe refusal) pairs, continuing from the previous round's adapter
if given.

Usage: python train_round.py --round 1 --prev-adapter checkpoints/round0-adapter \
    --harvested results/round0_harvested.json --out checkpoints/round1-adapter
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import config


def build_dataset(harvested_path: Path, tokenizer):
    records = json.load(open(harvested_path))
    texts = []
    for r in records:
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["safe_response"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return Dataset.from_dict({"text": texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--prev-adapter", type=str, default=None)
    ap.add_argument("--harvested", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config.DEFENDER_BASE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.DEFENDER_BASE_PATH, dtype=torch.bfloat16, device_map="cuda"
    )

    if args.prev_adapter:
        # merge previous round's adapter into base weights, then attach a
        # fresh LoRA for this round -- keeps adapter rank fixed across
        # rounds while still being cumulative.
        model = PeftModel.from_pretrained(model, args.prev_adapter)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(Path(args.harvested), tokenizer)

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=f"{args.out}-tmp",
        num_train_epochs=config.TRAIN_EPOCHS_PER_ROUND,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.TRAIN_GRAD_ACCUM,
        learning_rate=config.TRAIN_LR,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
