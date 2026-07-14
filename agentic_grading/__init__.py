"""rate_llm_answers: label bias in agentic LLM graders.

A blinded best-of-K rating experiment that probes label bias in
*agentic* graders (Claude Code, Codex, Qwen-Agent). All candidate
answers are authored by a single held-constant source model
(claude-haiku-4.5); only the fictional `model` label the grader sees
varies across slots, and it is permuted randomly per (seed, question).
"Bias" = a grader systematically preferring a particular label.

Two tasks are supported (JANEKD_TASK / --task):
  - alpaca   : subjective helpfulness grading of Alpaca instructions.
  - proofnet : rigor grading of undergraduate proofs whose statements
               come from the ProofNet benchmark.

Chain-of-thought is captured per backend: Claude extended thinking
(MAX_THINKING_TOKENS), Codex reasoning summaries
(model_reasoning_summary), and Qwen reasoning_content (best-effort).

Entry points:
  - python -m agentic_grading.orchestrate --task {alpaca,proofnet}
        end-to-end: smoke gate -> 3 graders -> CoT extract/judge/aggregate.
  - python -m agentic_grading.run --experiment {alpaca,proofnet}
        rate trials for one backend (JANEKD_RATER_BACKEND).
  - python -m agentic_grading.analyze --run-dir <dir>
        per-label / per-source pick-rate analysis for a run dir.
"""
