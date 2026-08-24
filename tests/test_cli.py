from click.testing import CliRunner
import pytest

from clawbench.cli import SCENARIO_CHOICES, cli, normalize_gateway_url
from clawbench.schemas import ScenarioDomain


def test_cli_scenario_choices_track_schema_enum():
    assert SCENARIO_CHOICES == [scenario.value for scenario in ScenarioDomain]


def test_run_command_forwards_judge_score_gate(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeResult:
        submission_id = "submission-1"

        def model_dump(self):
            return {"submission_id": self.submission_id}

    class FakeHarness:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return FakeResult()

    monkeypatch.setattr("clawbench.cli.BenchmarkHarness", FakeHarness)

    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "anthropic/claude-sonnet-4-6",
            "--judge-model",
            "judge-model",
            "--judge-affects-score",
            "--runs",
            "1",
            "--task",
            "t1-bugfix-discount",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["judge_model"] == "judge-model"
    assert captured["judge_affects_score"] is True
    assert output.read_text(encoding="utf-8")


class _CapturingHarness:
    """Records the kwargs the CLI builds without running a benchmark."""

    captured: dict[str, object] = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)

    async def run(self):
        class FakeResult:
            submission_id = "submission-1"

            def model_dump(self):
                return {"submission_id": self.submission_id}

        return FakeResult()


def _invoke_run(monkeypatch, tmp_path, extra_args: list[str]):
    monkeypatch.setattr("clawbench.cli.BenchmarkHarness", _CapturingHarness)
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "anthropic/claude-sonnet-4-6",
            "--runs",
            "1",
            "--task",
            "t1-bugfix-discount",
            "--output",
            str(output),
            *extra_args,
        ],
    )
    assert result.exit_code == 0, result.output
    return _CapturingHarness.captured


def test_run_command_defaults_to_local_gateway(monkeypatch, tmp_path):
    captured = _invoke_run(monkeypatch, tmp_path, [])

    assert captured["gateway_config"].url == "ws://localhost:18789"


def test_run_command_accepts_explicit_gateway_url(monkeypatch, tmp_path):
    """The harness measures an already-running gateway, so its URL is a knob.

    Without this the CLI could only ever measure a gateway on
    localhost:18789, which makes benchmarking a specific instance, host, or
    port impossible.
    """
    captured = _invoke_run(
        monkeypatch,
        tmp_path,
        ["--gateway-url", "ws://10.0.0.5:9999", "--gateway-token", "secret-token"],
    )

    assert captured["gateway_config"].url == "ws://10.0.0.5:9999"
    assert captured["gateway_config"].token == "secret-token"


def test_run_command_reads_gateway_url_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18800")

    captured = _invoke_run(monkeypatch, tmp_path, [])

    assert captured["gateway_config"].url == "ws://127.0.0.1:18800"


def test_explicit_gateway_url_beats_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18800")

    captured = _invoke_run(monkeypatch, tmp_path, ["--gateway-url", "ws://127.0.0.1:19999"])

    assert captured["gateway_config"].url == "ws://127.0.0.1:19999"


def test_empty_gateway_url_falls_back_to_default(monkeypatch, tmp_path):
    """An unset or blank env var must not produce an empty URL."""
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "")

    captured = _invoke_run(monkeypatch, tmp_path, [])

    assert captured["gateway_config"].url == "ws://localhost:18789"


def test_gateway_url_strips_shell_quoting(monkeypatch, tmp_path):
    """Leaked shell quotes otherwise become part of the hostname.

    That surfaces as a DNS gaierror, which looks like a firewall or network
    problem rather than a quoting typo.
    """
    captured = _invoke_run(
        monkeypatch, tmp_path, ["--gateway-url", "'ws://10.0.0.5:18789'"]
    )

    assert captured["gateway_config"].url == "ws://10.0.0.5:18789"


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://10.0.0.5:18789",
        "10.0.0.5:18789",
        "ws://'10.0.0.5':18789",
    ],
)
def test_gateway_url_rejects_unusable_values(monkeypatch, tmp_path, bad_url):
    monkeypatch.setattr("clawbench.cli.BenchmarkHarness", _CapturingHarness)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--model",
            "anthropic/claude-sonnet-4-6",
            "--runs",
            "1",
            "--task",
            "t1-bugfix-discount",
            "--output",
            str(tmp_path / "r.json"),
            "--gateway-url",
            bad_url,
        ],
    )

    assert result.exit_code != 0
    assert "gateway URL" in result.output


def test_normalize_gateway_url_accepts_valid_forms():
    assert normalize_gateway_url("ws://host:1") == "ws://host:1"
    assert normalize_gateway_url("  wss://host:1  ") == "wss://host:1"
    assert normalize_gateway_url("") == ""
