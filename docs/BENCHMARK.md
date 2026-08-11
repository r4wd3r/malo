# Benchmarking and Fair Evaluation

This document describes how MaLO workflows are configured for fair comparison and how effectiveness, efficiency, and halting cause are defined and logged.

## Unified configuration

All experiment YAMLs use the same values for comparable runs:

- **Model:** Same model for all agents in the workflow.
- **Temperature:** Same value for all agents (e.g. `0.2`).
- **Limits:** Same across experiments:
  - `max_turns`: 50
  - `observation_truncation_chars`: 5000
  - `recursion_limit`: 2000
  - `max_workers`: 3 (orchestrator architectures)
  - `worker_timeout_seconds`: e.g. 300 (orchestrator architectures only; wall-clock limit per worker)

No separate “benchmark” profile or config path is used; adjust the existing
architecture YAMLs under `config/<architecture>/architecture/` to keep these
values aligned.

## Minimal shared prompt kernel

Each workflow loads prompts from the YAMLs under `config/<architecture>/prompts/...`.
For fair comparison, every prompt file shares the same kernel wording:

- **Objective:** Injected via the `{objective}` placeholder (challenge, goal, context).
- **Execution environment:** Identical wording: agents/workers run with **high privileges**, can execute any command; all commands and scripts must be **non-interactive and complete** (no stdin prompts, no hanging waits; use flags such as `-y` for apt/pip where needed).
- **Tools:** Where an agent has tool access, the same bullet list and short descriptions are used (e.g. execute_script, submit_goal, give_up, request_human_help, and task_complete for workers).

Agent-specific instructions (ReAct loop, JSON output, critic verdict, worker report format) remain per workflow; only the kernel paragraphs above are unified.

## Metrics and halting

### Effectiveness

- **Success:** A run is successful when the workflow ends with `termination_reason == "success"` (goal met).
- **Failed:** Run ended without success (e.g. give_up, turn_limit, terminated).
- Success rate and per-run success are reported in CLI output and in batch JSON.

### Efficiency

- **Turns:** `turn_count` (main-loop turns: agent, reasoner, or orchestrator turns).
- **Tokens:** `total_tokens` (and optionally `prompt_tokens`, `completion_tokens`) from the final state.
- **Cost:** `cost_usd` is computed after each run from final-state token counts and the primary model’s pricing (see `observability/metrics.estimate_cost`). Batch JSON includes per-run `cost_usd` and a batch-level `total_cost_usd`.

### Halting cause

`termination_reason` in the final state (and in batch results) indicates why the run stopped. Possible values:

| Value          | Meaning                                    |
|----------------|--------------------------------------------|
| `success`      | Goal met (submit_goal succeeded).          |
| `give_up`      | Agent explicitly gave up.                  |
| `turn_limit`   | Stopped due to `max_turns` (turn budget).  |
| `terminated`   | Stopped by tool or internal termination.   |
| `no_messages`  | No messages to process (single-agent).     |
| `llm_timeout`    | LLM call timed out (agent/reasoner/orchestrator). |
| `tool_timeout`   | Tool execution timed out (e.g., execute_script). |
| `challenge_timeout` | Overall challenge timeout exceeded.      |
| `missing_goal`   | Goal validation not available.             |
| `error`          | Exception or runtime error.                |

Worker subgraph timeouts are reported in the worker report and logs; they do not
set the final `termination_reason`.

When the graph routes to end because `check_turn_limit(state, max_turns)` is true, the state is set to `termination_reason = "turn_limit"` before ending so that batch and CLI logs reflect this cause consistently.

## Logging

- **CLI (single run):** Final success/failure, termination reason, turn count, and cost are printed; metrics can be sent to WandB when enabled.
- **Batch run:** `BatchResult.to_dict()` and the saved JSON include for each challenge: `success`, `termination_reason`, `turn_count`, `total_tokens`, `cost_usd`, and `error` (if any). The batch also includes `total_cost_usd` (sum of per-run `cost_usd`).
