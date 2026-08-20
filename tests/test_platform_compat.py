"""Behavioral tests for cross-platform process and command helpers.

These exercise real subprocesses rather than asserting on implementation
details, because the failure they guard against -- a teardown that appears to
work while leaking grandchildren -- is invisible to a structural test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from clawbench.platform_compat import (
    IS_WINDOWS,
    default_temp_root,
    posix_shell_executable,
    resolve_command,
    resolve_executable,
    shell_command_argv,
    spawn_in_process_group,
    terminate_process_tree,
    user_state_dir,
)


def _pid_alive(pid: int) -> bool:
    if IS_WINDOWS:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
        )
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_resolve_executable_finds_interpreter_on_path() -> None:
    assert resolve_executable(sys.executable) is not None
    resolved = resolve_executable("python") or resolve_executable("python3")
    if resolved is not None:
        assert Path(resolved).exists()


def test_resolve_executable_returns_none_for_missing_command() -> None:
    assert resolve_executable("clawbench-definitely-not-a-real-binary") is None


def test_resolve_command_preserves_arguments() -> None:
    argv = resolve_command([sys.executable, "-c", "pass", "--flag"])
    assert argv[1:] == ["-c", "pass", "--flag"]


def test_resolve_command_leaves_unresolvable_command_untouched() -> None:
    # Popen should still be the thing that raises FileNotFoundError, so an
    # unknown command must survive resolution unchanged.
    argv = resolve_command(["clawbench-definitely-not-a-real-binary", "--x"])
    assert argv == ["clawbench-definitely-not-a-real-binary", "--x"]


@pytest.mark.skipif(not IS_WINDOWS, reason="PATHEXT shim resolution is Windows-specific")
def test_resolve_command_launches_windows_shim() -> None:
    """A bare .cmd/.bat name is not launchable by subprocess without PATHEXT."""
    shim = resolve_executable("npm")
    if shim is None:
        pytest.skip("npm is not installed on this machine")
    assert Path(shim).suffix.lower() in {".cmd", ".bat", ".exe"}
    completed = subprocess.run(
        resolve_command(["npm", "--version"]), capture_output=True, text=True
    )
    assert completed.returncode == 0


def test_terminate_process_tree_reaps_grandchildren(tmp_path: Path) -> None:
    """Teardown must kill the whole tree, not just the direct child.

    The gateway spawns Chromium; a leaked browser process would contaminate
    later runs, so this asserts on the grandchild actually being gone.
    """
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )

    process = spawn_in_process_group(
        [sys.executable, str(parent_script)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        grandchild_pid = int(process.stdout.readline().strip())
        assert _pid_alive(grandchild_pid)

        terminate_process_tree(process)

        deadline = time.time() + 15
        while time.time() < deadline and _pid_alive(grandchild_pid):
            time.sleep(0.25)
        assert not _pid_alive(grandchild_pid), "grandchild survived teardown"
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()


def test_terminate_process_tree_is_safe_on_exited_process() -> None:
    process = spawn_in_process_group([sys.executable, "-c", "pass"])
    process.wait(timeout=30)
    terminate_process_tree(process)


def test_terminate_process_tree_accepts_none() -> None:
    terminate_process_tree(None)


def test_shell_command_argv_preserves_posix_quoting() -> None:
    """Single quotes must stay quotes on every platform.

    cmd.exe treats ' as a literal character, so a benchmark check like
    `cat 'report 2026.json'` would silently mean something different on
    Windows and stop being comparable with the Linux cell.
    """
    completed = subprocess.run(
        shell_command_argv("echo 'report 2026.json'"),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "report 2026.json"


def test_shell_command_argv_treats_metacharacters_consistently(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    completed = subprocess.run(
        shell_command_argv("printf '%s' 'safe; touch injected'"),
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == "safe; touch injected"
    assert not marker.exists()


def test_spawn_in_process_group_runs_and_reports_exit_code() -> None:
    process = spawn_in_process_group([sys.executable, "-c", "raise SystemExit(7)"])
    assert process.wait(timeout=30) == 7


def test_default_temp_root_exists() -> None:
    assert default_temp_root().is_dir()


@pytest.mark.skipif(IS_WINDOWS, reason="guards the Linux baseline specifically")
def test_posix_keeps_literal_tmp_root() -> None:
    """The Linux cell is the study baseline, so its paths must not move.

    tempfile.gettempdir() follows TMPDIR and would relocate these in containers.
    """
    assert default_temp_root() == Path("/tmp")


@pytest.mark.skipif(IS_WINDOWS, reason="guards the Linux baseline specifically")
def test_posix_shell_is_bin_sh() -> None:
    """`shell=True` uses /bin/sh; preferring bash would upgrade fixture semantics."""
    assert posix_shell_executable() == "/bin/sh"


@pytest.mark.skipif(not IS_WINDOWS, reason="WSL shells only exist on Windows")
def test_windows_shell_is_never_wsl() -> None:
    """WSL's bash.exe runs commands in Linux, not Windows.

    Which bash wins is a PATH-ordering accident, so accepting a WSL one would
    make the operating system a benchmark run actually measures vary per
    machine while still reporting itself as native Windows.
    """
    shell = posix_shell_executable()
    if shell is None:
        pytest.skip("no POSIX shell installed on this machine")
    shell_dir = os.path.normcase(os.path.normpath(os.path.dirname(shell)))
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    forbidden = {
        os.path.normcase(os.path.normpath(os.path.join(system_root, "System32"))),
        os.path.normcase(os.path.normpath(os.path.join(system_root, "Sysnative"))),
    }
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        forbidden.add(
            os.path.normcase(
                os.path.normpath(os.path.join(local_app_data, "Microsoft", "WindowsApps"))
            )
        )
    assert shell_dir not in forbidden


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows path execution is the Windows failure mode")
def test_windows_shell_can_execute_a_windows_path() -> None:
    """The selected shell must be able to run the interpreter by its Windows path.

    This is the property WSL bash fails with
    `/bin/bash: line 1: E:\\...\\python.exe: command not found`, which surfaced
    as unrelated-looking execution-check failures.
    """
    if posix_shell_executable() is None:
        pytest.skip("no POSIX shell installed on this machine")
    completed = subprocess.run(
        shell_command_argv(f"'{sys.executable}' -c 'print(42)'"),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "42"


@pytest.mark.skipif(not IS_WINDOWS, reason="WSL shells only exist on Windows")
def test_wsl_bash_is_rejected_even_when_first_on_path(monkeypatch, tmp_path: Path) -> None:
    """Reproduces the reported failure: WSL bash winning the PATH lookup."""
    import clawbench.platform_compat as pc

    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    wsl_bash = Path(system_root) / "System32" / "bash.exe"
    if not wsl_bash.exists():
        pytest.skip("WSL bash is not installed on this machine")

    monkeypatch.setattr(pc, "_CACHED_WINDOWS_SHELL", None)
    monkeypatch.setenv("PATH", str(wsl_bash.parent))
    monkeypatch.delenv("CLAWBENCH_POSIX_SHELL", raising=False)

    selected = pc.posix_shell_executable()
    assert selected is None or Path(selected) != wsl_bash


@pytest.mark.skipif(not IS_WINDOWS, reason="override is only consulted on Windows")
def test_posix_shell_override_is_honoured(monkeypatch) -> None:
    """The matrix study needs to pin one shell across every cell."""
    import clawbench.platform_compat as pc

    baseline = pc.posix_shell_executable()
    if baseline is None:
        pytest.skip("no POSIX shell installed on this machine")

    monkeypatch.setattr(pc, "_CACHED_WINDOWS_SHELL", None)
    monkeypatch.setenv("CLAWBENCH_POSIX_SHELL", baseline)
    assert pc.posix_shell_executable() == baseline


@pytest.mark.skipif(IS_WINDOWS, reason="guards the Linux baseline specifically")
def test_resolve_command_is_a_noop_on_posix() -> None:
    """execvp resolves PATH after the child applies cwd; pre-resolving changes that."""
    assert resolve_command(["python3", "-c", "pass"]) == ["python3", "-c", "pass"]


def test_user_state_dir_matches_platform_convention() -> None:
    resolved = user_state_dir("openclaw")
    if IS_WINDOWS:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            assert resolved == Path(local_app_data) / "openclaw"
    else:
        # The Linux cell is the study baseline; state must stay at ~/.openclaw.
        assert resolved == Path.home() / ".openclaw"
