# Related work

## Datasets / benchmarks reused

- **AdvBench** (Zou, Wang, Kolter, Fredrikson, "Universal and Transferable
  Adversarial Attacks on Aligned Language Models", 2023 -- the GCG paper).
  520 (goal, target) harmful-behavior pairs. **What we took:** the raw CSV,
  stratified 40 train / 20 held-out with a fixed seed. **What we did
  ourselves:** the train/held-out split, the query-budget/PAIR loop around
  it, and the harvesting-into-SFT-data pipeline -- none of that exists in
  the original repo, which is attack-only.
- **HarmBench** (Mazeika et al., NeurIPS 2024 Datasets & Benchmarks) and
  **JailbreakBench** (Chao et al., NeurIPS 2024 D&B) -- standardized
  automated red-teaming harnesses with curated behavior sets and scoring
  protocols. **What we took:** the general idea of a fixed behavior set +
  a calibrated judge rubric rather than an ad hoc one. **What we did
  ourselves:** we did not adopt their exact harness/codebase; our judge
  (`judge.py`) is a hand-written StrongREJECT-style rubric prompt against a
  local Ollama model, not their released classifier.
- **StrongREJECT** (Souly et al., 2024) -- rubric-based jailbreak judging
  with reported human-level agreement. **What we took:** the framing of
  judging *degree of harmful compliance* (1-10) rather than a binary
  refused/complied label, which is what our `judge.py` rubric implements.
  We did not use their released judge model/weights.

## Attack methods

- **PAIR** (Chao et al., "Jailbreaking Black Box LLMs in Twenty Queries",
  2023) -- iterative black-box attacker that rewrites its jailbreak prompt
  using the target's last response, bounded query budget. **What we took:**
  the core loop shape (attacker sees behavior + prior prompt/response/score,
  escalates) implemented in `attacker.py`. **What we did ourselves:** our
  own prompt template, our own scoring-in-the-loop-history mechanism, and
  running it against a local Ollama model rather than their original
  target/attacker pairing.
- **GCG** (Zou et al., 2023) and **AutoDAN** (Liu et al., "Generating
  Stealthy Jailbreak Prompts on Aligned LLMs", 2023) -- token-level
  gradient search and genetic-algorithm-based suffix/prompt search,
  respectively. We did **not** implement either; we chose the LLM-red-teamer
  option from the assignment's two allowed defaults instead, since a
  white-box search re-run every round against a LoRA-shifting defender is
  more fragile to set up correctly than an in-context PAIR loop, given our
  time budget.

## Co-evolutionary loop papers (closest prior art to this whole track)

- **"Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via
  Tree-Group Dual-Aware Search and Optimization"** (Li, Song, Zhu, Chen,
  Tang, arXiv 2511.19218, 2025). Tree-search attacker + iterative defender
  realignment in a closed loop; closest published analogue to our Phase 1
  loop. We did not reuse their codebase; our loop is a much simpler
  PAIR-style attacker (no tree search) since our compute budget is a single
  shared GPU and 3-5 rounds, not a multi-week attacker-search infrastructure.
- **MAGIC** (arXiv 2602.01539, 2026) -- multi-turn multi-agent RL where an
  attacker agent learns to rewrite queries and a defender agent
  simultaneously updates via policy optimization. Directly relevant
  **template for our Phase 3 (bonus RL attacker)**, not used in Phase 1.
- **Self-RedTeam** (arXiv, 2026) -- online self-play RL, attacker and
  defender co-evolve under a game-theoretic objective. Same relevance as
  MAGIC: informs Phase 3 design, not implemented in Phase 1.
- **"Learning to Attack and Defend: Adaptive Red Teaming of Language Models
  via GRPO"** (arXiv 2606.09701) -- trains the attacker with GRPO (no
  learned value network, cheaper than PPO). **Planned reuse:** if we do
  attempt the Phase 3 RL bonus, GRPO (or a reward-weighted/RAFT-style
  simplification of it) is the practical choice over PPO given our shared
  single-GPU box.
- **"Short-length Adversarial Training Helps LLMs Defend Long-length
  Jailbreak Attacks"** (Fu, Ding, Zhang, Wang, NeurIPS 2025) -- theoretical
  and empirical result that training on short adversarial examples
  generalizes to longer ones. Relevant to why LoRA SFT on harvested
  (typically short, single-turn) jailbreak prompts should be expected to
  generalize to some degree, and something we can check against our
  held-out-behavior ASR results.
- **"Fight Back Against Jailbreaking via Prompt Adversarial Tuning"** (Mo et
  al., NeurIPS 2024) -- defends by learning a defensive prompt prefix
  instead of updating model weights. Not implemented (we update weights via
  LoRA per the assignment's requirement to "continue fine-tuning
  cumulatively"), but a useful contrast to discuss in the write-up's
  "what we rejected and why" section: a prompt-tuning defense would be
  cheaper per round but arguably violates the spirit of "the defender
  hardens," since a discovered jailbreak could route around a fixed prefix
  more easily than around updated weights.

## Refusal mechanism / attacker-model note

- **Arditi et al., "Refusal in Language Models Is Mediated by a Single
  Direction"** (2024) -- the mechanistic-interpretability result behind
  "abliteration". Directly informed our fix in `attacker_model_note.md`:
  swapping the frozen attacker from stock `qwen3.8:27b` to the abliterated
  `orcarouter/Qwen3.8-27B-Uncensored`, since a normally-aligned model
  refuses the red-teaming meta-task itself for top-severity behaviors.
