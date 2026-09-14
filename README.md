# Jailbreak Attack-Defense Co-Evolutionary Loop

**Track 2 submission.** A small, fixed-round adversarial self-play loop between a frozen LLM red-teamer and a genuinely fine-tuned defender, plus two follow-on experiments: a rehearsal (replay) ablation that fixes a catastrophic-forgetting failure in the naive design, and a bonus phase where the attacker itself becomes a small trainable RL policy.

- **Full write-up:** [`docs/writeup.md`](docs/writeup.md) (methodology, results, related work, rejected approaches, honest limitations)
- **Rendered report with charts:** published artifact (research-paper layout, real ASR/query-cost figures) -- ask for the link if you don't have it, or open `docs/writeup.md` for the same content in plain markdown
- **Results table:** [`results/RESULTS.md`](results/RESULTS.md)
- **Citations:** [`docs/related_work.md`](docs/related_work.md)
- **PDF version:** to be added (research-paper PDF export of the write-up)

## Trained artifacts on Hugging Face

**Adapters:** https://huggingface.co/Sanyam0605/jailbreak-coevo-adapters/tree/main (public)

| Path in repo | What it is |
|---|---|
| `round1-adapter`, `round2-adapter`, `round3-adapter` | Core-loop defender, no-rehearsal branch, rounds 1-3 (each continues cumulatively from the previous round) |
| `round1-adapter-rehearsal`, `round2-adapter-rehearsal`, `round3-adapter-rehearsal` | Core-loop defender, rehearsal branch -- fresh LoRA from base each round, trained on the union of all harvested attacks so far |
| `attacker-rl-round1`, `attacker-rl-round2` | RL bonus: the trainable attacker policy (base: `huihui-ai/Huihui-Qwen3.5-2B-abliterated`) after 1 and 2 rounds of best-of-K reward-weighted fine-tuning |
| `defender-rl-round1` | RL bonus: the defender after being hardened on round 1's RL-attacker successes |

Each folder is a standard PEFT LoRA adapter (`adapter_config.json` + `adapter_model.safetensors` + tokenizer files) -- load with `PeftModel.from_pretrained(base_model, "<local-or-hub-path>/<folder>")`. Core-loop / rehearsal adapters merge onto `LiquidAI/LFM2.5-1.2B-Instruct`; the RL-attacker adapters merge onto `huihui-ai/Huihui-Qwen3.5-2B-abliterated`; `defender-rl-round1` merges onto `LiquidAI/LFM2.5-1.2B-Instruct`.

**Datasets:** https://huggingface.co/datasets/Sanyam0605/jailbreak-coevo-data (public)

| Path in repo | What it is |
|---|---|
| `behaviors_train.jsonl`, `behaviors_heldout.jsonl` | Our stratified 40/20 split of AdvBench (seed 0) plus coarse category tags -- the split itself is created here, not upstream |
| `harvested_by_round/*.json` | Every round's (jailbreak prompt, generated safe refusal) pairs actually used to train each adapter above -- one file per round per branch |
| `RESULTS.md` | Copy of the results table for convenience |

We did not re-upload the raw AdvBench / XSTest source CSVs (external benchmarks, cited and linked instead -- see Setup below to fetch them directly from source).

## Model roster

| Role | Model | Trained? | Served via |
|---|---|---|---|
| Defender | `LiquidAI/LFM2.5-1.2B-Instruct` | Yes -- cumulative LoRA SFT each round | HF `transformers`, local GPU |
| Attacker (core loop, Phase 1) | `orcarouter/Qwen3.8-27B-Uncensored` | No -- frozen, in-context escalation only | Ollama, local |
| Attacker (RL bonus, Phase 3) | `huihui-ai/Huihui-Qwen3.5-2B-abliterated` | Yes -- RAFT-style best-of-K fine-tuning each round | HF `transformers`, local GPU |
| Judge | `gemma4:e4b` | No -- frozen, different family from either attacker | Ollama, local |

Why two different attackers: the core loop (default track) requires a frozen attacker mechanism; the RL bonus requires a *trainable* attacker policy, which must be loaded via `transformers`/`peft` (Ollama serves GGUF, inference-only). Why abliterated models for both attackers: a normally safety-tuned model refuses the red-teaming meta-task itself (writing a jailbreak) -- confirmed directly for the stock `qwen3.8:27b` and again for the defender's own base model, `LFM2.5-1.2B-Instruct`, when tested as an attacker. See [`docs/attacker_model_note.md`](docs/attacker_model_note.md).

## Setup

Everything runs on a single local GPU machine (this was run on an NVIDIA DGX Spark / GB10) with:
- [Ollama](https://ollama.com) running locally, with `orcarouter/Qwen3.8-27B-Uncensored` and `gemma4:e4b` pulled.
- Python 3.12 venv with: `torch` (CUDA build matching your GPU), `transformers>=5.12`, `peft`, `accelerate`, `trl`, `datasets`, `huggingface_hub`, `requests`.
- HF CLI logged in (`hf auth login`) to download the defender and RL-attacker base checkpoints.

```bash
uv venv .venv && source .venv/bin/activate   # or use any existing venv with the deps above
uv pip install transformers peft accelerate trl datasets huggingface_hub requests

# base models
hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir checkpoints/round0-base
hf download huihui-ai/Huihui-Qwen3.5-2B-abliterated --local-dir checkpoints/attacker-rl-base
ollama pull orcarouter/Qwen3.8-27B-Uncensored
ollama pull gemma4:e4b

# our trained adapters (optional -- skip if you're retraining from scratch)
hf download Sanyam0605/jailbreak-coevo-adapters --local-dir checkpoints/from-hub

# seed data
curl -sL "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv" -o data/advbench_harmful_behaviors.csv
curl -sL "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv" -o data/xstest_prompts.csv
python scripts/sample_behaviors.py        # writes data/behaviors_train.jsonl (40) + behaviors_heldout.jsonl (20)
python scripts/categorize_behaviors.py    # tags each behavior with a coarse category
```

## Running the core loop (Phase 1a: no rehearsal)

```bash
cd scripts
# round 0: attack the untouched base model
python run_round.py --round 0 --tag train --behaviors ../data/behaviors_train.jsonl --out-dir ../results
python eval_overrefusal.py --round 0 --n 50 --out-dir ../results

# round N: train on round (N-1)'s harvest, then attack the new defender
python train_round.py --round 1 --harvested ../results/round0_train_harvested.json --out ../checkpoints/round1-adapter
python run_round.py --round 1 --tag train --adapter ../checkpoints/round1-adapter --behaviors ../data/behaviors_train.jsonl --out-dir ../results
python run_round.py --round 1 --tag heldout --adapter ../checkpoints/round1-adapter --behaviors ../data/behaviors_heldout.jsonl --out-dir ../results
python eval_overrefusal.py --round 1 --adapter ../checkpoints/round1-adapter --n 50 --out-dir ../results
python analyze_round.py --round 1 --tag train   # ASR@7/9/10 + per-category breakdown

# repeat train_round.py -> run_round.py for rounds 2, 3, ...
```

`run_round.py` checkpoints incrementally per-behavior to `results/round{N}_{tag}_state.json` and resumes from it automatically if interrupted (e.g. an Ollama read-timeout on the 27B attacker) -- just rerun the same command.

**Important:** always pass a distinct `--tag` for different behavior sets at the same `--round` (`train` vs `heldout`) -- output filenames are `round{N}_{tag}_*`, and two runs with the same round+tag will silently overwrite each other's harvested data.

## Running the rehearsal variant (Phase 1b)

Same attack-data lineage as Phase 1a (reuses `round{0,1,2}_train_harvested.json`), only the training procedure differs: trains a fresh LoRA from the untouched base model each round on the *union* of all harvested sets so far, instead of continuing from the previous round's merged adapter on only the latest batch.

```bash
python train_round_rehearsal.py --round 2 \
    --harvested ../results/round0_train_harvested.json ../results/round1_train_harvested.json \
    --out ../checkpoints/round2-adapter-rehearsal
python run_round.py --round 2 --tag rehearsal --adapter ../checkpoints/round2-adapter-rehearsal \
    --behaviors ../data/behaviors_train.jsonl --out-dir ../results
```

## Running the RL bonus (Phase 3)

```bash
# round 1: untrained RL attacker vs. round-0 base defender
python rl_round.py --round 1 --n-behaviors 12 --out-attacker-adapter ../checkpoints/attacker-rl-round1 --out-dir ../results

# update the defender on round 1's successful jailbreaks (the other half of co-evolution)
python rl_harvest_to_defender_data.py --records ../results/rl_round1_records.json --out ../results/rl_round1_harvested.json
python train_round.py --round 1 --harvested ../results/rl_round1_harvested.json --out ../checkpoints/defender-rl-round1

# round 2: RL-trained attacker vs. the now-hardened defender
python rl_round.py --round 2 --n-behaviors 12 \
    --defender-adapter ../checkpoints/defender-rl-round1 \
    --prev-attacker-adapter ../checkpoints/attacker-rl-round1 \
    --out-attacker-adapter ../checkpoints/attacker-rl-round2 --out-dir ../results
```

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full table and diagnosis, and [`docs/writeup.md`](docs/writeup.md) for the complete analysis. Headline findings:

- **No-rehearsal core loop**: ASR stays pinned near ceiling across 3 rounds (1.000 -> 0.975 -> 1.000 -> 1.000 at threshold 7); the sub-metrics that do move (strict ASR@10, query cost, per-category ASR) oscillate rather than trend down. A round-1 gain on self-harm prompts (ASR 1.00 -> 0.67) fully reverts by round 2 -- a catastrophic-forgetting-style failure, since each round only trains on its own newest harvest.
- **Rehearsal variant**: identical attack data, only the training procedure changed (replay all harvested sets so far, not just the newest) -- produces a clean, monotonic hardening trend instead: strict ASR@10 falls from 0.825 to 0.600 by round 2, and a round-3 sample (n=11) shows the steepest drop yet (ASR 45.5%).
- **RL bonus (2 rounds)**: the trainable attacker's one round of RAFT fine-tuning fully offset the defender's one round of hardening -- ASR held exactly flat (0.583 -> 0.583) while the average judge score rose slightly, a small "arms race" result.

## Repo layout

```
scripts/      all training/attack/eval code
data/         AdvBench (train/heldout split), XSTest, category tags
checkpoints/  LoRA adapters + base model snapshots (mirrored on HF, see above)
results/      every round's raw attempts, harvested data, and summaries
docs/         writeup.md (full report), attacker_model_note.md, related_work.md, plan.md
```

## Key scripts

| Script | Purpose |
|---|---|
| `sample_behaviors.py` | Stratifies AdvBench into the fixed 40/20 train/held-out split |
| `categorize_behaviors.py` | Coarse keyword taxonomy tagging (cybercrime, weapons/CBRN, fraud, self-harm, hate, misinformation, other) |
| `attacker.py` / `ollama_client.py` | Frozen PAIR-style red-teamer against the local Ollama attacker model |
| `defender.py` | Loads the defender base model + optional LoRA adapter, generates responses |
| `judge.py` | 1-10 harm-compliance scoring via a separate frozen Ollama model |
| `run_round.py` | One round of attack + harvest, with incremental checkpointing/resume |
| `train_round.py` | Cumulative LoRA SFT, merges the previous round's adapter before training a fresh one |
| `train_round_rehearsal.py` | Rehearsal variant: fresh LoRA from base, trained on the union of all harvested rounds |
| `analyze_round.py` | Post-hoc ASR@7/9/10 and per-category breakdown from already-logged attempts |
| `eval_overrefusal.py` | XSTest-based over-refusal probe |
| `rl_attacker.py` / `rl_round.py` | Trainable RL attacker policy + one round of sample/score/RAFT-train |
| `rl_harvest_to_defender_data.py` | Converts RL-round attack records into defender training data |
