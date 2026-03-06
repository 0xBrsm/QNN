#!/usr/bin/env python3
"""Manage a detached corpus-crawl worker.

The worker runs canonical NetQuake crawl passes in a loop and logs output.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _daemonize() -> None:
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.setsid()

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, "r", encoding="utf-8") as devnull_r:
        os.dup2(devnull_r.fileno(), 0)
    with open(os.devnull, "a", encoding="utf-8") as devnull_w:
        os.dup2(devnull_w.fileno(), 1)
        os.dup2(devnull_w.fileno(), 2)


def start(args: argparse.Namespace) -> int:
    pid_path = Path(args.pid_file)
    log_path = Path(args.log_file)

    existing_pid = _read_pid(pid_path)
    if existing_pid and _is_pid_running(existing_pid):
        print(f"worker already running pid={existing_pid}")
        return 0

    _daemonize()

    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    running = True

    def _handle_term(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    base_cmd = [
        sys.executable,
        "src/scripts/fetch_netquake_corpus.py",
        "--focus-profile",
        args.focus_profile,
        "--out",
        args.out,
        "--remote-share",
        args.remote_share,
        "--remote-username",
        args.remote_username,
        "--remote-password",
        args.remote_password,
        "--remote-free-threshold",
        str(args.remote_free_threshold),
        "--space-check-every",
        str(args.space_check_every),
        "--max-pages",
        str(args.max_pages),
        "--max-downloads",
        str(args.max_downloads),
        "--max-total-gb",
        str(args.max_total_gb),
        "--sleep",
        str(args.sleep),
    ]

    _write_line(log_path, f"[{_timestamp()}] worker start pid={os.getpid()}")

    while running:
        _write_line(log_path, f"[{_timestamp()}] pass start")
        with log_path.open("a", encoding="utf-8") as log_handle:
            proc = subprocess.run(base_cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        _write_line(log_path, f"[{_timestamp()}] pass end rc={proc.returncode}")

        if not running:
            break
        time.sleep(max(args.loop_sleep, 1.0))

    _write_line(log_path, f"[{_timestamp()}] worker stopping")
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass
    return 0


def stop(args: argparse.Namespace) -> int:
    pid_path = Path(args.pid_file)
    pid = _read_pid(pid_path)
    if not pid:
        print("worker not running (no pid file)")
        return 0
    if not _is_pid_running(pid):
        print(f"stale pid file found pid={pid}; removing")
        pid_path.unlink(missing_ok=True)
        return 0

    os.kill(pid, signal.SIGTERM)
    timeout = time.time() + 10
    while time.time() < timeout:
        if not _is_pid_running(pid):
            break
        time.sleep(0.2)

    if _is_pid_running(pid):
        os.kill(pid, signal.SIGKILL)

    pid_path.unlink(missing_ok=True)
    print(f"worker stopped pid={pid}")
    return 0


def status(args: argparse.Namespace) -> int:
    pid_path = Path(args.pid_file)
    pid = _read_pid(pid_path)
    if pid and _is_pid_running(pid):
        print(f"worker running pid={pid}")
        return 0
    print("worker not running")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detached NetQuake corpus worker")
    parser.add_argument("action", choices=["start", "stop", "status"])

    parser.add_argument("--pid-file", default="../artifacts/corpus/netquake/meta/corpus_worker.pid")
    parser.add_argument("--log-file", default="../artifacts/corpus/netquake/meta/corpus_worker.log")

    parser.add_argument("--focus-profile", default="canonical-netquake", choices=["all", "canonical-netquake"])
    parser.add_argument("--out", default="../artifacts/corpus/netquake")
    parser.add_argument("--remote-share", default=r"\\pi.local\nqcorpus\netquake")
    parser.add_argument("--remote-username", default="guest")
    parser.add_argument("--remote-password", default="guest")
    parser.add_argument("--remote-free-threshold", type=float, default=0.10)
    parser.add_argument("--space-check-every", type=int, default=25)
    parser.add_argument("--max-pages", type=int, default=10000)
    parser.add_argument("--max-downloads", type=int, default=600000)
    parser.add_argument("--max-total-gb", type=float, default=300.0)
    parser.add_argument("--sleep", type=float, default=0.001)
    parser.add_argument("--loop-sleep", type=float, default=30.0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.action == "start":
        return start(args)
    if args.action == "stop":
        return stop(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
