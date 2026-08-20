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

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"

# Job handles are keyed by pid so teardown can find the job that owns a child
# without threading an extra object through every call site.
_JOB_HANDLES: dict[int, Any] = {}

# Shell discovery runs a verification subprocess, so the answer is cached.
# "" means "searched and found nothing".
_CACHED_WINDOWS_SHELL: str | None = None


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
    """Resolve argv[0] to a concrete executable path on Windows only.

    POSIX is deliberately left untouched. `execvp` already performs PATH lookup
    correctly there, and crucially it does so *after* the child applies `cwd`;
    resolving in the parent instead would change which executable a relative
    PATH entry selects. Windows needs the pre-resolution because it has no
    PATHEXT handling in `subprocess`.
    """
    argv = list(command)
    if not IS_WINDOWS or not argv:
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


def _normalized_dir(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.dirname(path)))


def _wsl_shell_dirs() -> set[str]:
    """Directories whose `bash.exe` is really a WSL launcher, not a shell.

    Running benchmark commands through these would execute them inside a Linux
    VM while the harness believes it is measuring native Windows -- the whole
    cell would silently be measuring the wrong operating system.
    """
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(system_root, "System32"),
        os.path.join(system_root, "Sysnative"),
    ]
    if local_app_data:
        candidates.append(os.path.join(local_app_data, "Microsoft", "WindowsApps"))
    return {os.path.normcase(os.path.normpath(item)) for item in candidates}


def _is_wsl_shell(path: str) -> bool:
    return _normalized_dir(path) in _wsl_shell_dirs()


def _windows_shell_candidates() -> list[str]:
    """Git for Windows / MSYS2 bash locations, most standard first."""
    roots: list[str] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = os.environ.get(env_name)
        if base:
            roots.append(os.path.join(base, "Git"))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(os.path.join(local_app_data, "Programs", "Git"))

    # A Git install that ships bash keeps it beside git.exe's parent.
    git_exe = shutil.which("git")
    if git_exe:
        roots.append(os.path.dirname(os.path.dirname(git_exe)))

    candidates: list[str] = []
    for root in roots:
        for relative in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
            candidates.append(os.path.join(root, *relative))
    return candidates


def _shell_handles_windows_paths(shell: str) -> bool:
    """Confirm `shell` can execute a native Windows path.

    This is the property that actually matters and the one a WSL bash fails, so
    it is checked directly rather than inferred from the shell's location.
    """
    try:
        completed = subprocess.run(
            [shell, "-c", f"'{sys.executable}' -c 'print(1)'"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "1"


def posix_shell_executable() -> str | None:
    """Locate the shell used to run benchmark shell commands.

    On POSIX this is always ``/bin/sh`` -- the shell ``subprocess(shell=True)``
    uses. Preferring ``bash`` here would silently upgrade fixture semantics on
    the Linux baseline cell, letting bashisms start working and changing what
    the benchmark measures.

    On Windows there is no ``/bin/sh``, so a native POSIX shell (Git for
    Windows / MSYS2 bash) is used to keep POSIX quoting semantics intact. WSL's
    ``bash.exe`` is deliberately rejected: it is a different operating system,
    and whether it wins depends on PATH ordering, so accepting it would make the
    measured OS a per-machine accident.
    """
    if not IS_WINDOWS:
        return "/bin/sh"

    global _CACHED_WINDOWS_SHELL
    if _CACHED_WINDOWS_SHELL is not None:
        return _CACHED_WINDOWS_SHELL or None

    override = os.environ.get("CLAWBENCH_POSIX_SHELL")
    ordered: list[str] = []
    if override:
        ordered.append(override)
    for name in ("bash", "sh"):
        found = shutil.which(name)
        if found:
            ordered.append(found)
    ordered.extend(_windows_shell_candidates())

    seen: set[str] = set()
    verified_rejects: list[str] = []
    for candidate in ordered:
        key = os.path.normcase(os.path.normpath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if not os.path.exists(candidate):
            continue
        if _is_wsl_shell(candidate):
            verified_rejects.append(candidate)
            continue
        if not _shell_handles_windows_paths(candidate):
            verified_rejects.append(candidate)
            continue
        _CACHED_WINDOWS_SHELL = candidate
        return candidate

    _CACHED_WINDOWS_SHELL = ""
    logger.debug("No usable POSIX shell found; rejected: %s", verified_rejects)
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
            "No usable POSIX shell was found for running benchmark shell commands.\n"
            "Install Git for Windows, which provides a native bash.exe, or set "
            "CLAWBENCH_POSIX_SHELL to one.\n"
            "Note that WSL's bash.exe (under System32 or WindowsApps) is rejected "
            "on purpose: it runs commands inside Linux, so a 'native Windows' run "
            "would silently measure a different operating system."
        )
    return [shell, "-c", command]


def default_temp_root() -> Path:
    """Platform-appropriate replacement for hardcoded /tmp.

    POSIX keeps ``/tmp`` literally so the Linux baseline's paths -- and any
    tooling that reads them -- are unchanged; ``tempfile.gettempdir()`` would
    follow ``TMPDIR`` and silently move them in containers.
    """
    if not IS_WINDOWS:
        return Path("/tmp")
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
