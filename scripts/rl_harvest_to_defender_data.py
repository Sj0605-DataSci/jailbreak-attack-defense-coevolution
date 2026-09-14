"""Convert an rl_round.py records file (behavior, prompt, score) into the
(prompt, safe_response) harvested format train_round.py expects, so the
defender can be updated on the RL attacker's successful jailbreaks -- the
other half of genuine co-evolution (see rl_round.py's docstring).
"""
import argparse
import json
from pathlib import Path

import config
from run_round import write_safe_response

SUCCESS_THRESHOLD = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    records = json.load(open(args.records))
    harvested = []
    for r in records:
        if r["score"] >= SUCCESS_THRESHOLD:
            safe_response = write_safe_response(r["prompt"], r["behavior"])
            harvested.append({"prompt": r["prompt"], "safe_response": safe_response})
    json.dump(harvested, open(args.out, "w"), indent=2)
    print(f"wrote {len(harvested)} harvested examples to {args.out}")


if __name__ == "__main__":
    main()
