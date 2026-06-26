"""Client for the resident BC ablation daemon."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

from qnn.bc.ablation_daemon import DEFAULT_SOCKET


def _send(socket_path: str, request: dict[str, Any]) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        raw = sock.makefile("rb").readline()
    if not raw:
        raise RuntimeError("daemon closed connection without a response")
    response = json.loads(raw.decode("utf-8"))
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "daemon request failed"))
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Control the resident BC ablation daemon")
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Unix socket path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="Queue one or more run dirs")
    p_submit.add_argument("run_dirs", nargs="+", type=Path)

    sub.add_parser("status", help="Show daemon state")

    p_parallel = sub.add_parser("set-parallel", help="Set max concurrently running ablations")
    p_parallel.add_argument("parallel_runs", type=int)

    p_cancel = sub.add_parser("cancel", help="Cancel a queued job")
    p_cancel.add_argument("job_id")

    sub.add_parser("reset", help="Unload the resident source bundle while idle")
    sub.add_parser(
        "reload-code",
        help="Hot-reload BC code in-process while idle, keeping the VRAM corpus "
        "(rebuilds it only if the dequant data layer changed)",
    )
    sub.add_parser("shutdown", help="Stop the daemon")

    args = parser.parse_args()
    if args.cmd == "submit":
        request = {"cmd": "submit", "run_dirs": [str(path) for path in args.run_dirs]}
    elif args.cmd == "status":
        request = {"cmd": "status"}
    elif args.cmd == "set-parallel":
        request = {"cmd": "set_parallel", "parallel_runs": args.parallel_runs}
    elif args.cmd == "cancel":
        request = {"cmd": "cancel", "job_id": args.job_id}
    elif args.cmd == "reset":
        request = {"cmd": "reset"}
    elif args.cmd == "reload-code":
        request = {"cmd": "reload_code"}
    elif args.cmd == "shutdown":
        request = {"cmd": "shutdown"}
    else:  # pragma: no cover - argparse enforces this.
        raise RuntimeError(f"unsupported command: {args.cmd}")

    response = _send(args.socket, request)
    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
