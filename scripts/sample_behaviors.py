"""Subsample AdvBench into a fixed, reproducible set of behaviors for the loop.

We don't run all 520 AdvBench behaviors per round (too slow for 3-5 rounds x
multi-query PAIR attacks on a single GPU). Instead we fix a stratified random
subset once, split into train (attacked every round, harvested for SFT) and
held-out eval (used only to measure generalization of the defender's learned
refusals, never harvested).
"""
import csv
import random
import json
from pathlib import Path

SEED = 0
N_TRAIN = 40
N_HELDOUT = 20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    rows = list(csv.DictReader(open(DATA_DIR / "advbench_harmful_behaviors.csv")))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    train = rows[:N_TRAIN]
    heldout = rows[N_TRAIN:N_TRAIN + N_HELDOUT]

    with open(DATA_DIR / "behaviors_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps({"behavior": r["goal"], "target": r["target"]}) + "\n")

    with open(DATA_DIR / "behaviors_heldout.jsonl", "w") as f:
        for r in heldout:
            f.write(json.dumps({"behavior": r["goal"], "target": r["target"]}) + "\n")

    print(f"wrote {len(train)} train behaviors, {len(heldout)} heldout behaviors")


if __name__ == "__main__":
    main()
