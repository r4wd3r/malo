# MaLO

<img width="1536" height="1024" alt="MaLO GitHub Banner" src="https://github.com/user-attachments/assets/f851156b-daf0-47e9-84f3-b44c19f149d4" />

**Multi-agent LLM Operations for Autonomous Security.** Code for the paper *"MaLO: Multi-agent LLM Operations for Autonomous Security"*: The first systematic study of how agent topology affects autonomous cyber operations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/IEEE%20CNS-2026-b31b1b.svg)](#)
[![Award](https://img.shields.io/badge/🏆%20Award-USCYBERCOM-gold.svg)](#)

> 🏆 **Awarded by U.S. Cyber Command (USCYBERCOM) at CyberRecon 2025 (Hunter's Award) and 2026 (Best Presentation Award)** for earlier versions of this work.


## Paper
[IEEE Conference on Communications and Network Security (CNS) 2026](https://ieeexplore.ieee.org/abstract/document/11720875)

Citation:
```
@INPROCEEDINGS{castro2026_malo,
  author={Castro, Sebastián R. and Taranenko, Iakov and Tsai, Astra and Lau, Nancy and Cardenas, Alvaro A.},
  booktitle={2026 IEEE Conference on Communications and Network Security (CNS)}, 
  title={MaLO: Multi-agent LLM Operations for Autonomous Security}, 
  year={2026},
  pages={1-9},
  keywords={Modeling;Security;Complexity theory;Cognition;Cognitive systems;Tools;Large language models;Turning;Architecture;Computer architecture;autonomous cyber operations;multi-agent systems;large language models;security benchmarking;agent topology},
  doi={10.1109/CNS69863.2026.11720875}}

```

## Why MaLO?

Most multi-agent security frameworks hardcode one pipeline and call it a day. MaLO treats **agent topology as a first-class variable**: swap between a single ReAct agent, a reasoner/executor split, or an orchestrator dispatching parallel workers (with or without a critic), and benchmark them head-to-head on identical tasks, tools, and prompts.

We used it to solve tasks up to **85x faster** than a human security expert and hit a **78.6%** solve rate on our 42-task benchmark. Full results in the paper.

## Features

Configuration-driven: define architecture, model, and challenge in YAML, and MaLO handles graph construction and Docker lifecycle.

- **Four architectures** — see below and `docs/WORKFLOWS.md`.
- **Goal validation**: exact_match, list_match, regex.
- **Challenge formats**: MaLO benchmark (`challenge.yml` + `src/`) and NYU CTF.
- **Batch mode**: run many challenges under one architecture/experiment config; results as JSON with success rate and cost.
- **Docker**: ephemeral containers via `EnvironmentManager`; optional `--skip-env`.
- **Observability**: token/cost tracking, WandB logging when enabled.

## Architectures

| Name | What it does |
|---|---|
| `single_react` | One agent, ReAct-style reasoning + tool calls. |
| `reasoner_executor` | Reasoning and execution split across two agents. |
| `orchestrator_workers` | An orchestrator plans and dispatches parallel worker agents. |
| `orchestrator_critic_workers` | `orchestrator_workers`, plus a critic that reviews plans before execution. |

See `docs/WORKFLOWS.md` for full flow diagrams and turn/timeout limits.

## Challenges

MaLO uses **MaLO-format** challenges: a `challenge.yml` plus a `src/` directory. See `docs/GOAL_VALIDATION.md` for goal formats and `docs/BENCHMARK.md` for benchmark rules.

## Installation

```bash
git clone --recurse-submodules git@github.com:<this_repo>
cd MaLO
./scripts/setup.sh
```

See docs/SETUP.md for full prerequisites and setup details.

## Quick Start

Single challenge:
`poetry run malo run --arch single_react -c path/to/challenge`

Batch run:
`poetry run malo run --arch single_react --batch benchmarks/malo-benchmark \
  --output results.json`

Customize runs by editing YAML configs or using CLI overrides like --model and --provider. See docs/CONFIG.md and docs/CLI.md.

## CLI

Use poetry run for CLI commands to ensure the project environment is active.

Common commands:
```
- poetry run malo list
- poetry run malo run --arch single_react -c <challenge>
- poetry run malo run --config <path> --batch <dir> --output results.json
```

See docs/CLI.md for full CLI and layering details.

## Configuration

See docs/CONFIG.md for YAML structure, provider settings, and examples.

Documentation

- docs/SETUP.md — prerequisites and install/setup steps
- docs/CONFIG.md — YAML configuration and providers
- docs/CLI.md — CLI and config layering
- docs/WORKFLOWS.md — architecture flows and limits
- docs/BENCHMARK.md — fair evaluation and metrics
- docs/CONTRIBUTING.md — contributor guidelines
- docs/GOAL_VALIDATION.md — goal configuration and validation
