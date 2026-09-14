# Plan

## Phase 1 -- Core co-evolutionary loop (default track, in progress)

- Defender: LiquidAI/LFM2.5-1.2B-Instruct, cumulative LoRA SFT each round.
- Attacker: frozen `orcarouter/Qwen3.8-27B-Uncensored` (abliterated Qwen3.8-27B),
  PAIR-style, in-context escalation only, never fine-tuned. See
  `attacker_model_note.md` for why the stock (safety-tuned) qwen3.8:27b
  didn't work as an attacker.
- Judge: frozen `gemma4:e4b` (different family from attacker, avoids
  attacker/judge collusion).
- Loop: attack round N's defender on 40 train behaviors (AdvBench subset,
  seed 0) -> harvest successful jailbreaks + contextual safe refusals ->
  LoRA SFT round N+1 defender from round N's adapter -> repeat, 3-5 rounds.
- Metrics per round: ASR, avg queries used to succeed, held-out ASR (20
  behaviors never harvested, generalization check), and a small benign-prompt
  set to watch for over-refusal drift.

## Phase 2 -- Scaling study across defender model sizes

Once the core loop is validated on LFM2.5-1.2B, rerun the identical loop
(same attacker, judge, behaviors, round count) on a small family of
differently-sized open-weight instruct defenders to see how base capability/
alignment interacts with the co-evolution dynamics -- does a bigger base
model start more robust, converge faster, or show different degenerate
failure modes (over-refusal, mode collapse)? Candidates (need to fit
comfortably alongside the attacker+judge on the shared GB10):
- LiquidAI/LFM2.5-1.2B-Instruct (phase 1 baseline)
- A second Liquid or Qwen3.5 small size (e.g. Qwen3.5-2B) for a same-family
  size comparison
- Optionally one different-family model (e.g. a Llama-3.2 or SmolLM2 size)
  to separate "size" effects from "family/alignment-recipe" effects

Keep everything else fixed (behaviors, attacker, judge, LoRA rank/hparams,
round count) so ASR-vs-round curves are comparable across sizes -- this is
the actual "scaling law" plot: x-axis defender size, y-axis
round-0-ASR / round-N-ASR / rounds-to-halve-ASR, etc.

## Phase 3 -- Bonus: RL co-trained attacker

Only after phase 1 (and ideally phase 2) are done and documented. Initialize
a small instruct model (e.g. the same size class as the defender, or smaller)
as the attacker's own policy and update it against the *current* round's
defender each round -- both models now genuinely co-evolve. Given compute,
scope to reward-weighted fine-tuning (simpler, cheaper) over full PPO unless
time allows: sample K candidate attacks per behavior from the attacker
policy, score with the judge, fine-tune the attacker on its own
highest-reward samples (best-of-K distillation / RAFT-style loop), then
fine-tune the defender as in phase 1 on what got through. Clearly document
how many rounds this was actually affordable for, since it doubles the
per-round training cost.
