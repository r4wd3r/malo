# Configuration

This document describes YAML configuration, provider settings, and examples.

## Architecture Configs

Canonical architecture YAMLs live under:

```
config/<architecture>/architecture/<architecture>.yaml
```

Prompts live under:

```
config/<architecture>/prompts/
```

## Experiments Workspace

- `config/` holds canonical templates and prompts.
- `experiments/` is a workspace for per-run variants and traces.

## Limits and Defaults

Experiment YAML selects **architecture** and **limits** (e.g. `max_turns`,
`observation_truncation_chars`, `max_workers`, `worker_max_turns`,
`max_message_chars`, `random_seed`). Use the same model, temperature, and
limits across runs for fair comparison (see `docs/BENCHMARK.md`).

## LLM Provider Configuration

Each agent can set `provider` and optional `base_url` in YAML, which enables
mixed backends in one session.

**IMPORTANT**: API keys must be set as environment variables. They cannot be
specified in YAML files.

### YAML Configuration (Recommended)

OpenRouter:

```yaml
agents:
  agent:
    model: "google/gemini-3-flash-preview"
    temperature: 0.2
    prompts_file: "../prompts/agent.yaml"
    provider: "openrouter"
```

Ollama:

```yaml
agents:
  agent:
    model: "llama3.2"
    temperature: 0.2
    prompts_file: "../prompts/agent.yaml"
    provider: "ollama"
    base_url: "http://localhost:11434/v1"  # optional
```

Mixed backends:

```yaml
agents:
  orchestrator:
    model: "google/gemini-3-flash-preview"
    temperature: 0.2
    prompts_file: "../prompts/orchestrator.yaml"
    provider: "openrouter"
  worker:
    model: "llama3.2"
    temperature: 0.2
    prompts_file: "../prompts/worker.yaml"
    provider: "ollama"
```

### Environment Variables

OpenRouter:
- `LLM_PROVIDER`: defaults to `openrouter`.
- `OPENROUTER_API_KEY`: required for OpenRouter.
- `OPENROUTER_BASE_URL`: optional override for OpenRouter endpoint.

Ollama:
- `LLM_PROVIDER`: set to `ollama`.
- `OLLAMA_BASE_URL`: optional override (default `http://localhost:11434/v1`).

**Model format**:
- OpenRouter: provider/model (e.g., `google/gemini-3-flash-preview`).
- Ollama: local model name (e.g., `llama3.2`, `mistral`).

## Example Architecture Paths

- `config/single_react/architecture/single_react.yaml`
- `config/reasoner_executor/architecture/reasoner_executor.yaml`
- `config/orchestrator_workers/architecture/orchestrator_workers.yaml`
- `config/orchestrator_critic_workers/architecture/orchestrator_critic_workers.yaml`
