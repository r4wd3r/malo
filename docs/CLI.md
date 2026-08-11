# MaLO CLI

This document describes the CLI and configuration layering model for researcher
workflows. It prioritizes fast iteration while preserving full configuration
granularity.

## Goals

- Make common runs short and repeatable.
- Keep all workflow/graph configuration in YAML with override layering.
- Allow adding new workflows without CLI changes.
- Preserve current behavior where possible.

## Non-goals

- Hide or reduce configuration expressiveness.
- Introduce implicit file mutations or auto-saves.

## Command Overview

`malo run`
- Primary command for single or batch runs.
- Supports config layering and CLI overrides.

`malo list`
- Lists available architecture configs from `config/**/architecture/*.yaml`.
- Output includes the config path and a short alias derived from filename.

`malo config show`
- Resolves config with layering and prints the final merged config.

`malo config validate`
- Validates a config file (or layered config) and prints errors.

## Config Sources

Base config sources:
- `--config <path>`: explicit YAML file.
- `--arch <name>`: resolves to `config/<name>/architecture/<name>.yaml`.

Overrides:
- `--overrides <path>`: explicit YAML files, applied in order.
- `--set key=value`: final overrides using dotted paths.

## Precedence Rules

1. Base config from `--config` or `--arch`.
2. `--overrides` files in the order provided.
3. `--set` key-value overrides.

Later inputs override earlier ones.

## Merge Semantics

- Dictionaries are deep-merged.
- Lists are replaced in full.
- Unknown keys are rejected by the Pydantic model (no implicit extension).

## `malo run` Flags

Required:
- Either `--config` or `--arch`.

Run target:
- `-c`, `--challenge <path>`: single challenge folder.
- `--batch <path>`: batch mode over a directory.

Config layering:
- `--overrides <path>`: repeatable.
- `--set key=value`: repeatable.

Convenience overrides:
- `--model <name>`: apply to all agents present in config.
- `--provider <name>`: apply to all agents present in config.

Execution:
- `--output <path>`: batch results JSON.
- `--skip-env`: skip Docker.
- `--no-wandb`: disable WandB.
- `--verbose`, `--quiet`: output control.

## `--model` and `--provider` Semantics

- `--model` sets `agents.*.model` for every agent that exists in the loaded
  config. This overrides model values from base config and overrides.
- `--provider` sets `agents.*.provider` for every agent that exists in the
  loaded config. This overrides provider values from base config and overrides.
- If a config omits a given agent, no value is injected for that agent.

## Examples

List available architectures:

```
malo list
```

Single challenge with a base architecture and overrides:

```
malo run --arch single_react -c path/to/challenge \
  --overrides experiments/my_run/architecture/limits_fast.yaml \
  --set limits.max_turns=20
```

Batch run with global model override:

```
malo run --arch orchestrator_workers --batch benchmarks/malo-benchmark \
  --model google/gemini-3-flash-preview --output results.json
```

## Experiments vs Config

- `config/` holds canonical architecture templates and prompts.
- `experiments/` is a workspace for per-run variants and traces.
- The CLI only enumerates architecture YAMLs under `config/`.
