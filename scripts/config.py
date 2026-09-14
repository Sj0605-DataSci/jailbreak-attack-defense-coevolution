from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFENDER_BASE_PATH = ROOT / "checkpoints" / "round0-base"
CHECKPOINTS_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Frozen attacker: local strong model via Ollama, never fine-tuned.
# Uses an abliterated (refusal-removed) variant of the same base as the judge
# is built on, because a normally safety-tuned model refuses the red-teaming
# meta-task itself (writing a jailbreak) for top-severity categories -- see
# docs/attacker_model_note.md for the diagnosis that led to this swap.
ATTACKER_MODEL = "orcarouter/Qwen3.8-27B-Uncensored"
# Judge: a DIFFERENT frozen model than the attacker, to avoid collusion.
JUDGE_MODEL = "gemma4:e4b"

N_ROUNDS = 4
QUERIES_PER_BEHAVIOR = 5  # PAIR-style refinement attempts per behavior per round
MAX_NEW_TOKENS_DEFENDER = 256

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# LFM2 is a hybrid short-conv + gated-attention architecture (not a plain
# Llama-style transformer), so its Linear module names differ:
# attention blocks use q_proj/k_proj/v_proj/out_proj, conv blocks use
# in_proj/out_proj, and the SwiGLU MLP uses w1/w2/w3.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]

TRAIN_EPOCHS_PER_ROUND = 2
TRAIN_LR = 1e-4
TRAIN_BATCH_SIZE = 2
TRAIN_GRAD_ACCUM = 4
