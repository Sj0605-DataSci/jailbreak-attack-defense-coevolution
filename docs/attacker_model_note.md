# Why the attacker model is abliterated, not the stock instruct model

## What happened

First smoke test used `qwen3.8:27b` (stock, safety-tuned) as the frozen
red-teamer, per the "LLM red-teamer" default option in the assignment. On
the two AdvBench behaviors tested (suicide instructions, insider trading
guide), the attack success rate came back 0/2 across all 5 query attempts
each.

Inspecting `results/round999_attempts.json` showed the actual bug: the
attacker model refused the *meta-task* of writing a jailbreak prompt, and
its raw refusal text got treated as the "prompt" and sent straight to the
defender. E.g. for query 0 on the suicide behavior, the attacker's full
output was "I won't generate that prompt. Creating an optimized adversarial
prompt specifically designed to extract suicide instructions from an AI
model is not something I'll help with..." -- so the defender never actually
saw an attack, and both the defender and judge correctly rated the
(non-adversarial) exchange as safe. ASR=0 was a true negative on a
broken input, not evidence the defender is robust.

## Why this happens

This is a known failure mode in the jailbreak-research literature: a
strongly safety-tuned model applies its own refusal behavior recursively to
the task of *generating* jailbreaks, especially for top-severity categories
(self-harm, weapons, csam-adjacent). The original PAIR (Chao et al. 2023)
and AutoDAN (Liu et al. 2023) papers deliberately used weakly-aligned
attacker models (Vicuna, Mixtral-Instruct) for exactly this reason --
GPT-4-class models are poor attackers because they refuse the attacker
role itself.

## Fix

Swap the attacker model to `orcarouter/Qwen3.8-27B-Uncensored`, an
abliterated variant of the same Qwen3.8-27B base already used as the local
judge/defender-adjacent model on this box. Abliteration (Arditi et al.,
"Refusal in LLMs is mediated by a single direction", 2024) surgically
removes the refusal direction from the residual stream without otherwise
retraining the model, so capability is preserved and only the tendency to
refuse is gone. Using the same base model family as before keeps the
attacker's reasoning/writing quality comparable to the original choice --
only its willingness to produce jailbreak text changes.

The **judge** stays on the normally-aligned `gemma4:e4b` -- it doesn't need
to write harmful content, only to score whether the defender's response
was harmful, so no abliteration is needed or wanted there (we want the
judge's own safety-relevant judgment intact).
