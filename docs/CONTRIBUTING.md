# Contributing

## Principles
- Keep changes focused and minimize unrelated refactors.
- Follow the existing code style (Black + isort).
- Update docs and tests when behavior changes.

## Development Setup
```bash
poetry install
```

## Running Tests
```bash
poetry run pytest
```

## Formatting & Linting
```bash
poetry run black .
poetry run isort .
poetry run pylint MaLO
```

## Branching & PRs
- Use feature branches off `main`.
- Keep commit messages descriptive (`feat: ...`, `fix: ...`, `doc: ...`).
- PRs should include: summary, rationale, and tests run.

## Dependency Changes
- Runtime deps go under `[tool.poetry.dependencies]`.
- Dev-only deps go under `[tool.poetry.group.dev.dependencies]`.
