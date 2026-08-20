"""Cross-platform process spawning, teardown, and command resolution.

ClawBench spawns long-lived children (the OpenClaw gateway, task fixture
services) that themselves spawn grandchildren -- most importantly Chromium.
Leaking those between runs silently contaminates later measurements, so
teardown has to be tree-shaped on every platform.

This module is the single source of truth for that behaviour. Call sites use
`spawn_in_process_group()` and `terminate_process_tree()` rather than growing
their own platform branches; before this existed the POSIX ``killpg`` dance was
duplicated in two places and neither worked on Windows.

Platform notes:

* POSIX puts the child in a new session (``start_new_session``) and signals the
  whole process group.
* Windows has no process groups in the POSIX sense. ``start_new_session`` is
  *silently ignored* there, so the pre-existing teardown path looked correct
  while providing no containment at all. Instead the child is assigned to a Job
  Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, which the kernel tears
  down as a unit -- covering grandchildren even if the direct child has already
  exited.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

IS_WINDOWS = os.name == "nt"

# Job handles are keyed by pid so teardown can find the job that owns a child
# without threading an extra object through every call site.
_JOB_HANDLES: dict[int, Any] = {}


def resolve_executable(name: str) -> str | None:
    """Return an absolute path for `name`, honouring Windows PATHEXT.

    `subprocess` on Windows does not apply PATHEXT when launching a bare
    command, so ``Popen(["npm"])`` raises FileNotFoundError even when npm is on
    PATH as ``npm.CMD``. ``shutil.which`` does apply PATHEXT, so resolving first
    and launching the resolved path works for shims and real executables alike.
    """
    if not name:
        return None
    found = shutil.which(name)
    if found:
        return found
    # An explicit path that exists but is not on PATH is still launchable.
    candidate = Path(name)
    if candidate.exists():
        return str(candidate)
    return None


def resolve_command(command: Sequence[str]) -> list[str]:
    """Resolve argv[0] to a concrete executable path, leaving arguments intact."""
    argv = list(command)
    if not argv:
        return argv
    resolved = resolve_executable(argv[0])
    if resolved:
        argv[0] = resolved
    return argv


def process_group_spawn_kwargs() -> dict[str, Any]:
    """Popen kwargs that make a child independently killable as a tree."""
    if IS_WINDOWS:
        # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT for a graceful
        # stop; CREATE_SUSPENDED is deliberately not used so callers keep normal
        # Popen semantics. The job object assignment happens post-spawn.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _assign_to_job_object(process: subprocess.Popen) -> None:
    """Put `process` in a kill-on-close Job Object so its tree dies with it."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job_object_extended_limit_information = 9
        job_object_limit_kill_on_job_close = 0x00002000
        process_set_quota = 0x0100
        process_terminate = 0x0001

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
        if not kernel32.SetInformationJobObject(
            job,
            job_object_extended_limit_information,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(job)
            return

        handle = kernel32.OpenProcess(
            process_set_quota | process_terminate, False, process.pid
        )
        if not handle:
            kernel32.CloseHandle(job)
            return
        try:
            if not kernel32.AssignProcessToJobObject(job, handle):
                kernel32.CloseHandle(job)
                return
        finally:
            kernel32.CloseHandle(handle)

        _JOB_HANDLES[process.pid] = job
    except Exception:
        # Job objects are an optimisation for guaranteed cleanup; if they are
        # unavailable we still fall back to taskkill /T below.
        return


def spawn_in_process_group(command: Sequence[str], **kwargs: Any) -> subprocess.Popen:
    """Spawn `command` so that `terminate_process_tree` can reap its whole tree.

    Resolves argv[0] (so Windows shims work) unless the caller passes
    ``shell=True``, where the string is handed to the platform shell verbatim.
    """
    if kwargs.get("shell"):
        argv: Any = command
    else:
        argv = resolve_command(command)
    spawn_kwargs = {**process_group_spawn_kwargs(), **kwargs}
    process = subprocess.Popen(argv, **spawn_kwargs)
    _assign_to_job_object(process)
    return process


def _terminate_windows_tree(process: subprocess.Popen, *, force: bool) -> None:
    job = _JOB_HANDLES.get(process.pid)
    if job is not None and force:
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # Closing the kill-on-close job terminates every process in it.
            kernel32.TerminateJobObject(job, 1)
            kernel32.CloseHandle(job)
        except Exception:
            pass
        finally:
            _JOB_HANDLES.pop(process.pid, None)
        return

    if not force:
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except Exception:
            pass

    # Fall back to the built-in tree killer. /PID keeps this scoped to the
    # process we spawned; /T includes descendants.
    taskkill = resolve_executable("taskkill")
    if taskkill:
        try:
            subprocess.run(
                [taskkill, "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
            return
        except Exception:
            pass
    try:
        process.kill()
    except Exception:
        pass


def _signal_posix_group(process: subprocess.Popen, sig: int) -> None:
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, OSError):
        pass


def signal_process_tree(process: subprocess.Popen, *, force: bool) -> None:
    """Ask a spawned tree to stop (`force=False`) or kill it outright."""
    if IS_WINDOWS:
        _terminate_windows_tree(process, force=force)
        return
    _signal_posix_group(process, signal.SIGKILL if force else signal.SIGTERM)


def terminate_process_tree(
    process: subprocess.Popen | None,
    *,
    graceful_timeout: float = 5.0,
    force_timeout: float = 3.0,
) -> None:
    """Stop a process and everything it spawned, escalating if it lingers."""
    if process is None or process.poll() is not None:
        _release_job(process)
        return

    signal_process_tree(process, force=False)
    try:
        process.wait(timeout=graceful_timeout)
        _release_job(process)
        return
    except subprocess.TimeoutExpired:
        pass

    signal_process_tree(process, force=True)
    try:
        process.wait(timeout=force_timeout)
    except subprocess.TimeoutExpired:
        pass
    _release_job(process)


def _release_job(process: subprocess.Popen | None) -> None:
    if process is None or not IS_WINDOWS:
        return
    job = _JOB_HANDLES.pop(process.pid, None)
    if job is None:
        return
    try:
        import ctypes

        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
    except Exception:
        pass


def posix_shell_executable() -> str | None:
    """Locate a POSIX shell, including Git for Windows' bash."""
    for candidate in ("bash", "sh"):
        found = shutil.which(candidate)
        if found:
            return found
    if IS_WINDOWS:
        for fallback in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Git"
            / "bin"
            / "bash.exe",
        ):
            if fallback.exists():
                return str(fallback)
    return None


def shell_command_argv(command: str) -> list[str]:
    """Build an argv that runs a POSIX shell string on any platform.

    Task fixtures and execution checks are authored as POSIX shell (single-quote
    quoting, pipes, redirects). Running them through ``cmd.exe`` on Windows
    would change their meaning -- single quotes stop being quotes -- so the same
    POSIX shell semantics are used everywhere. Keeping the shell identical
    across cells is what makes cross-OS scores comparable at all.
    """
    shell = posix_shell_executable()
    if shell is None:
        raise RuntimeError(
            "A POSIX shell is required to run benchmark shell commands. "
            "On Windows install Git for Windows (provides bash.exe) or WSL."
        )
    return [shell, "-c", command]


def default_temp_root() -> Path:
    """Platform-appropriate replacement for hardcoded /tmp."""
    return Path(tempfile.gettempdir())


def user_state_dir(app: str = "openclaw") -> Path:
    """Resolve the per-user state directory for `app` on this platform.

    POSIX behaviour is deliberately left as ``~/.<app>`` rather than switching to
    XDG: the Linux cell is the study's baseline, so the port must not move where
    it stores state.
    """
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / app
    return Path.home() / f".{app}"


__all__ = [
    "IS_WINDOWS",
    "default_temp_root",
    "posix_shell_executable",
    "process_group_spawn_kwargs",
    "resolve_command",
    "resolve_executable",
    "shell_command_argv",
    "signal_process_tree",
    "spawn_in_process_group",
    "terminate_process_tree",
    "user_state_dir",
]
