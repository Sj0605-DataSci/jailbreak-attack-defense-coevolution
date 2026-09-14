"""Rehearsal-variant defender training: instead of continuing cumulatively
from the previous round's merged adapter using only that round's newest
harvested batch (train_round.py), train a FRESH LoRA from the untouched base
model each round on the UNION of all harvested examples from round 0..N.

Why: the no-rehearsal loop (train_round.py) showed an oscillation --
self_harm category ASR dropped 1.0->0.67 at round 1, then bounced back to
1.0 at round 2 -- consistent with catastrophic forgetting, since each
round's small (~40 example) LoRA has no pressure to retain earlier rounds'
narrow fixes. Training on the cumulative replay buffer each round is the
standard continual-learning fix for exactly this failure mode.

Deliberately starts from the clean base model each round (not the previous
rehearsal adapter) so rehearsal-round-N is a clean function of "all data up
to round N", not path-dependent on its own training history -- keeps the
comparison against the no-rehearsal condition interpretable.

Usage: python train_round_rehearsal.py --round 2 \
    --harvested ../results/round0_train_harvested.json ../results/round1_train_harvested.json ../results/round2_train_harvested.json \
    --out ../checkpoints/round2-adapter-rehearsal
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import config


def build_dataset(harvested_paths: list[Path], tokenizer):
    texts = []
    for path in harvested_paths:
        for r in json.load(open(path)):
            messages = [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["safe_response"]},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return Dataset.from_dict({"text": texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--harvested", type=str, nargs="+", required=True,
                     help="all harvested files from round 0 through this round, in order")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config.DEFENDER_BASE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.DEFENDER_BASE_PATH, dtype=torch.bfloat16, device_map="cuda"
    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset([Path(p) for p in args.harvested], tokenizer)
    print(f"rehearsal dataset size (union of {len(args.harvested)} rounds): {len(dataset)}")

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
    print(f"saved rehearsal adapter to {args.out}")


if __name__ == "__main__":
    main()
