"""Coarse keyword-based categorization of AdvBench behaviors into a
HarmBench/JailbreakBench-style taxonomy, so we can report per-category ASR
trends (does the attacker specialize / does one category resist longer)
instead of only an aggregate number.

This is a lightweight heuristic, not a learned classifier -- good enough to
spot gross category-level patterns across only 40 train behaviors.
"""
import json
from pathlib import Path

CATEGORY_KEYWORDS = {
    "cybercrime": ["hack", "malware", "virus", "exploit", "ransomware", "phishing",
                   "ddos", "botnet", "password", "network", "software"],
    "weapons_cbrn": ["bomb", "explosive", "weapon", "chemical", "biological", "nuclear",
                     "poison", "gun", "firearm"],
    "fraud_financial": ["insider trading", "launder", "counterfeit", "fraud", "scam",
                         "tax evasion", "embezzle", "steal"],
    "self_harm": ["suicide", "self-harm", "self harm"],
    "hate_harassment": ["racist", "hate", "harass", "bully", "discriminat"],
    "misinformation": ["fake news", "disinformation", "misinformation", "conspiracy",
                        "propaganda"],
}


def categorize(behavior: str) -> str:
    b = behavior.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in b for kw in keywords):
            return cat
    return "other"


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for split in ["behaviors_train.jsonl", "behaviors_heldout.jsonl"]:
        path = data_dir / split
        rows = [json.loads(l) for l in open(path)]
        for r in rows:
            r["category"] = categorize(r["behavior"])
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        print(split, Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
