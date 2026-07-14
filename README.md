<h1 align="center">Value Leakage: An LLM's Answers Are Silently Shaped by Its Own Values</h1>

<p align="center">
Jan Betley<sup>1,*</sup> &nbsp; Johannes Treutlein<sup>1,*</sup> &nbsp; Jan Dubiński<sup>1,2,3,†</sup> &nbsp; Harry Mayne<sup>1,†</sup> &nbsp; Karol Gałązka<sup>1</sup> &nbsp; Niels Warncke<sup>4</sup> &nbsp; Anna Sztyber-Betley<sup>3</sup> &nbsp; Owain Evans<sup>1</sup>
</p>

<p align="center">
<sup>1</sup>Truthful AI &nbsp; <sup>2</sup>NASK National Research Institute &nbsp; <sup>3</sup>Warsaw University of Technology &nbsp; <sup>4</sup>Center on Long-Term Risk
</p>

<p align="center">
<sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Work done during Astra Fellowship at Constellation, Berkeley
</p>

## Abstract

*People use language models for practical questions whose answers are difficult to
verify. We show that models exhibit covert value leakage: the information they
provide is influenced by their own values, without this influence being disclosed
to the user.*

*In one of our evaluations, the user is considering investing in an AI company
and wants to know how likely the AI bubble is to pop. Claude Opus 4.8 gives a
lower probability when the company under consideration is Anthropic rather than
OpenAI. Yet Claude mostly fails to disclose this influence to the user.*

*Covert value leakage is a form of misalignment because it goes against the user's
preferences and is likely to mislead them. To investigate this phenomenon, we
introduce a suite of evaluations to quantify value leakage and whether models
disclose it. We find that models are influenced by different types of values,
including preferences for morally good outcomes, for the company that developed
them, and for some human leisure activities over others.*

*We often observe large differences among frontier models on the same evaluation.
For example, on a Fermi-estimation task, Claude models falsely claim to give
unbiased answers in their chain-of-thought, while Qwen models explain how their
values bias their answers. Value leakage is a failure mode distinct from
sycophancy and reward hacking, and current alignment training and evaluations do
not adequately address it.*

## Setup

```bash
git clone --recurse-submodules https://github.com/TruthfulAI-research/value_leakage.git
cd value_leakage
uv sync
```

The cached rollouts and judge outputs used in the paper live in the
[value_leakage_data](https://github.com/TruthfulAI-research/value_leakage_data)
repository, mounted as the `data/` submodule (~8 GB). If you cloned without
`--recurse-submodules`, run `git submodule update --init` to fetch it. With the
data present, the analysis and plotting scripts run without API access;
sampling scripts re-query the models.

API keys are read from the environment / `.env` (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `DASHSCOPE_API_KEY`,
`TINKER_API_KEY`).

## Code by paper section

| Paper section | Experiment | Code |
|---|---|---|
| §3, App. C | Donation Bet | `donation_bet/` |
| §4, App. D | AI Bubble & AGI Tweet | `ai_company_questions/` |
| §5, App. E | Job Offer | `job_offer/` |
| §6, App. F | Agentic Grading | `agentic_grading/` |
| §7, App. G | Choosing Activities | `choosing_activities/` |
| App. H | Agentic Effort | TBD |

Shared infrastructure (model registry, API senders, caching, judges, plot
styling) is in `shared/`. Cached experiment data is in the `data/` submodule.
