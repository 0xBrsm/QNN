from __future__ import annotations

import json

from quake_ai import check_accelerator


def test_main_reports_runtime(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "quake_ai.check_accelerator.describe_torch_runtime",
        lambda requested: {"requested_device": requested, "resolved_device": "cpu", "backend": "cpu"},
    )
    monkeypatch.setattr("sys.argv", ["check_accelerator.py", "--device", "cpu"])

    assert check_accelerator.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_device"] == "cpu"
    assert payload["resolved_device"] == "cpu"


def test_main_fails_on_error_when_requested(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "quake_ai.check_accelerator.describe_torch_runtime",
        lambda requested: {"requested_device": requested, "resolved_device": "unavailable", "error": "no gpu"},
    )
    monkeypatch.setattr("sys.argv", ["check_accelerator.py", "--device", "gpu", "--fail-on-error"])

    assert check_accelerator.main() == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_device"] == "gpu"
    assert payload["error"] == "no gpu"
