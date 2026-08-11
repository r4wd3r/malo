from __future__ import annotations

from MaLO.graphs.output_models import CriticOutput, ReasonerOutput
from MaLO.graphs.utils import parse_typed_json


def test_parse_typed_json_reasoner_output() -> None:
    content = '{"thought": "t", "command": {"type": "bash", "script": "ls"}}'
    model, error = parse_typed_json(content, ReasonerOutput)
    assert error is None
    assert model is not None
    assert model.command.type == "bash"


def test_parse_typed_json_critic_verdict_normalizes() -> None:
    content = '{"verdict": "reject", "feedback": "no"}'
    model, error = parse_typed_json(content, CriticOutput)
    assert error is None
    assert model is not None
    assert model.verdict == "REJECT"
