"""Unix-socket daemon for resident BC/head-probe ablation sweeps.

Start this once inside the trainer container, then submit compatible run
dirs while it stays alive. The daemon keeps one source bundle resident and
trains queued jobs against it without rematerializing the corpus.
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qnn.bc.code_reload import (
    changed_data_layer,
    data_layer_fingerprint,
    reload_bc_code,
)


def _import_app_symbols() -> dict[str, Any]:
    """Import the BC entry points the daemon calls, as a name→object dict.

    Factored out so that :meth:`AblationDaemon._reload_code` can rebind these
    module globals after :func:`reload_bc_code` purges ``sys.modules`` — a
    top-level ``from x import name`` would otherwise keep pointing at the
    pre-reload objects. Called once at import and again after every reload.
    """
    from qnn.bc.container import (
        BCSourceBundle,
        build_behavior_cloning_sources,
        source_compatibility_key_for_config,
    )
    from qnn.bc.train import BCConfig, run_behavior_cloning
    from qnn.run.common import (
        RunnerContext,
        base_results,
        build_runner_context,
        finalize_results,
        prepare_bc_run_outputs,
    )
    from qnn.run.config import build_run_bc_config

    return {k: v for k, v in locals().items()}


globals().update(_import_app_symbols())


DEFAULT_SOCKET = "/tmp/qnn-bc-ablation.sock"


@dataclass(slots=True)
class _AblationJob:
    job_id: str
    ctx: RunnerContext
    config: BCConfig
    source_key: tuple[Any, ...]
    seed_checkpoint: str
    model_factory: Callable[[int, Any], Any] | None
    graph: Any | None
    side_channel_provider: Callable[..., Any] | None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_bc_config_dict(raw: dict[str, Any]) -> None:
    valid_keys = {f.name for f in _dc.fields(BCConfig)}
    unknown = sorted(set(raw) - valid_keys)
    if unknown:
        raise RuntimeError(
            f"BC config has {len(unknown)} unknown key(s) "
            f"(typo or removed feature): {unknown}. "
            "Either remove them from the run's train.json/model.json or add "
            "them to BCConfig."
        )


def _build_job(job_id: str, run_dir: Path) -> _AblationJob:
    ctx = build_runner_context(run_dir)
    if ctx.mode == "bc":
        raw_cfg = build_run_bc_config(ctx.run_cfg, ctx.device)
        _validate_bc_config_dict(raw_cfg)
        bc_config = BCConfig(**raw_cfg)
        model_factory = None
        graph = None
        side_channel_provider = None
    elif ctx.mode == "bench":
        from qnn.model.bench.runner import _build_bench_bc_config
        from qnn.model.bench.side_channels import bench_side_channel_scope

        bc_config, model_factory, graph = _build_bench_bc_config(ctx.run_cfg, ctx.device)
        side_channel_provider = bench_side_channel_scope
    else:
        raise RuntimeError(
            f"{ctx.run_dir}: mode {ctx.mode!r} cannot use the BC ablation "
            "daemon; expected 'bc' or 'bench'."
        )

    return _AblationJob(
        job_id=job_id,
        ctx=ctx,
        config=bc_config,
        source_key=source_compatibility_key_for_config(bc_config, graph=graph),
        seed_checkpoint=str(ctx.run_cfg.get("checkpoint_path", "") or ""),
        model_factory=model_factory,
        graph=graph,
        side_channel_provider=side_channel_provider,
    )


def _job_summary(job: _AblationJob, status: str) -> dict[str, Any]:
    payload = {
        "id": job.job_id,
        "status": status,
        "run_dir": str(job.ctx.run_dir),
        "run_name": str(job.ctx.manifest.get("name", job.ctx.run_dir.name)),
        "mode": job.ctx.mode,
        "submitted_at": job.submitted_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
    if job.error:
        payload["error"] = job.error
    return payload


def _job_log_label(job: _AblationJob) -> str:
    run_name = str(job.ctx.manifest.get("name", "") or job.ctx.run_dir.name)
    return f"{job.job_id}:{run_name}"


class AblationDaemon:
    def __init__(self, *, socket_path: Path, parallel_runs: int) -> None:
        self.socket_path = Path(socket_path)
        self.parallel_runs = max(1, int(parallel_runs))
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[_AblationJob] = []
        self._active: dict[str, _AblationJob] = {}
        self._done: list[_AblationJob] = []
        self._failed: list[_AblationJob] = []
        self._source_bundle: BCSourceBundle | None = None
        self._source_key: tuple[Any, ...] | None = None
        # Fingerprint of the data-baking sources at the moment the resident
        # bundle was built. A reload that finds these changed must drop the
        # bundle rather than serve stale (already-dequantized) tensors.
        self._data_layer_fp: str | None = None
        self._loading = False
        self._stopping = False
        self._server: socket.socket | None = None
        self._next_job_id = 1
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="bc-ablation-dispatch",
            daemon=True,
        )

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(16)
        server.settimeout(0.5)
        os.chmod(self.socket_path, 0o600)
        self._server = server
        self._dispatcher.start()
        print(
            f"  [bc/daemon] listening on {self.socket_path} "
            f"parallel_runs={self.parallel_runs}",
            flush=True,
        )

    def serve_forever(self) -> None:
        if self._server is None:
            self.start()
        assert self._server is not None
        while True:
            with self._lock:
                if self._stopping:
                    break
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                with self._lock:
                    if self._stopping:
                        break
                raise
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                name="bc-ablation-client",
                daemon=True,
            ).start()
        self._shutdown_server()

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass

    def _shutdown_server(self) -> None:
        self.stop()
        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=2.0)
        with self._lock:
            bundle = self._source_bundle
            self._source_bundle = None
            self._source_key = None
        if bundle is not None:
            bundle.release_device_tensors()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                raw = conn.makefile("rb").readline()
                if not raw:
                    return
                request = json.loads(raw.decode("utf-8"))
                response = self.handle_request(request)
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            conn.sendall((json.dumps(_json_safe(response)) + "\n").encode("utf-8"))

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = str(request.get("cmd", "")).strip()
        if cmd == "submit":
            paths = request.get("run_dirs", None)
            if paths is None:
                single = request.get("run_dir", None)
                paths = [single] if single is not None else []
            return self._submit([Path(str(p)) for p in paths])
        if cmd == "status":
            return {"ok": True, "state": self._status()}
        if cmd == "set_parallel":
            value = int(request["parallel_runs"])
            with self._cv:
                self.parallel_runs = max(1, value)
                self._cv.notify_all()
            return {"ok": True, "state": self._status()}
        if cmd == "cancel":
            return self._cancel(str(request["job_id"]))
        if cmd == "reset":
            return self._reset()
        if cmd == "reload_code":
            return self._reload_code()
        if cmd == "shutdown":
            self.stop()
            return {"ok": True, "state": self._status()}
        raise RuntimeError(f"unknown command: {cmd!r}")

    def _submit(self, run_dirs: list[Path]) -> dict[str, Any]:
        if not run_dirs:
            raise RuntimeError("submit requires at least one run directory")
        accepted: list[dict[str, Any]] = []
        with self._cv:
            for run_dir in run_dirs:
                job_id = f"job-{self._next_job_id:06d}"
                self._next_job_id += 1
                job = _build_job(job_id, run_dir)
                if self._source_key is None:
                    self._source_key = job.source_key
                elif job.source_key != self._source_key:
                    raise RuntimeError(
                        f"{run_dir}: incompatible source config for resident daemon. "
                        "Use reset after active/queued jobs finish, or start another daemon."
                    )
                self._queue.append(job)
                accepted.append(_job_summary(job, "queued"))
            self._cv.notify_all()
        return {"ok": True, "accepted": accepted, "state": self._status()}

    def _cancel(self, job_id: str) -> dict[str, Any]:
        with self._cv:
            for idx, job in enumerate(self._queue):
                if job.job_id == job_id:
                    job.finished_at = time.time()
                    job.error = "cancelled before start"
                    self._failed.append(job)
                    del self._queue[idx]
                    self._cv.notify_all()
                    return {"ok": True, "cancelled": _job_summary(job, "cancelled")}
            if job_id in self._active:
                job = self._active[job_id]
                job.cancel_event.set()
                return {"ok": True, "cancelled": _job_summary(job, "cancellation requested")}
        raise RuntimeError(f"{job_id} not found in queued or active jobs")

    def _reset(self) -> dict[str, Any]:
        with self._cv:
            if self._queue or self._active or self._loading:
                raise RuntimeError("reset requires daemon to be idle")
            bundle = self._source_bundle
            self._source_bundle = None
            self._source_key = None
            self._data_layer_fp = None
        if bundle is not None:
            bundle.release_device_tensors()
        return {"ok": True, "state": self._status()}

    def _reload_code(self) -> dict[str, Any]:
        """Hot-reload the BC code graph in-process, keeping the VRAM corpus.

        Idle-gated (like ``reset``): no job thread may be importing or running
        reloaded modules concurrently. The resident bundle is kept across the
        reload UNLESS the data-baking sources (see
        :func:`qnn.bc.code_reload.data_layer_fingerprint`) changed — in which
        case the already-dequantized tensors would be stale, so the bundle is
        released and the next submit rebuilds the corpus from fresh code.
        """
        with self._cv:
            if self._queue or self._active or self._loading:
                raise RuntimeError("reload-code requires daemon to be idle")
            stale = changed_data_layer(self._data_layer_fp)
            dropped_bundle = stale and self._source_bundle is not None
            bundle_to_release = self._source_bundle if dropped_bundle else None
            if dropped_bundle:
                self._source_bundle = None
                self._source_key = None
                self._data_layer_fp = None

            report = reload_bc_code()
            # Rebind this module's entry-point globals to the freshly imported
            # objects; the pre-reload `from x import name` bindings are stale.
            globals().update(_import_app_symbols())

        if bundle_to_release is not None:
            bundle_to_release.release_device_tensors()

        print(
            f"  [bc/daemon] reloaded {report['purged_count']} qnn modules; "
            f"bundle {'dropped (data layer changed)' if dropped_bundle else 'kept'}",
            flush=True,
        )
        return {
            "ok": True,
            "reloaded_modules": report["purged_count"],
            "data_layer_changed": stale,
            "bundle_dropped": dropped_bundle,
            "state": self._status(),
        }

    def _status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "parallel_runs": self.parallel_runs,
                "loading": self._loading,
                "has_source_bundle": self._source_bundle is not None,
                "source_key": _json_safe(self._source_key),
                "queued": [_job_summary(job, "queued") for job in self._queue],
                "active": [_job_summary(job, "active") for job in self._active.values()],
                "done": [_job_summary(job, "done") for job in self._done[-20:]],
                "failed": [_job_summary(job, "failed") for job in self._failed[-20:]],
                "stopping": self._stopping,
            }

    def _dispatch_loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: (
                        self._stopping
                        or (
                            bool(self._queue)
                            and len(self._active) < self.parallel_runs
                            and not self._loading
                        )
                    )
                )
                if self._stopping:
                    return
                if self._source_bundle is None:
                    first = self._queue[0]
                    self._loading = True
                else:
                    first = None

            if first is not None:
                try:
                    print(
                        f"  [bc/daemon] loading source bundle for {first.ctx.run_dir}",
                        flush=True,
                    )
                    bundle = build_behavior_cloning_sources(first.config, graph=first.graph)
                    loaded_error = None
                except Exception as exc:
                    bundle = None
                    loaded_error = exc
                    # Fail LOUDLY: a swallowed source-load error (e.g. a
                    # collection_fingerprint mismatch) reads as a silent hang
                    # to anyone watching `docker logs`. Emit the full
                    # traceback to stderr so the real reason is visible
                    # without having to query the daemon's _failed list.
                    print(
                        f"  [bc/daemon] ERROR: source load FAILED for "
                        f"{first.ctx.run_dir}\n{traceback.format_exc()}",
                        file=sys.stderr,
                        flush=True,
                    )
                with self._cv:
                    self._loading = False
                    if loaded_error is not None:
                        failed = self._queue.pop(0)
                        failed.finished_at = time.time()
                        failed.error = f"source load failed: {loaded_error}"
                        self._failed.append(failed)
                        if not self._queue and not self._active:
                            self._source_key = None
                        self._cv.notify_all()
                        continue
                    assert bundle is not None
                    self._source_bundle = bundle
                    self._source_key = bundle.compatibility_key
                    self._data_layer_fp = data_layer_fingerprint()
                    self._cv.notify_all()

            while True:
                with self._cv:
                    if (
                        self._stopping
                        or self._source_bundle is None
                        or not self._queue
                        or len(self._active) >= self.parallel_runs
                    ):
                        break
                    job = self._queue.pop(0)
                    self._active[job.job_id] = job
                    job.started_at = time.time()
                    bundle = self._source_bundle
                threading.Thread(
                    target=self._run_job,
                    args=(job, bundle),
                    name=f"bc-ablation-{job.job_id}",
                    daemon=True,
                ).start()

    def _run_job(self, job: _AblationJob, source_bundle: BCSourceBundle) -> None:
        print(f"  [bc/daemon] start {job.job_id}: {job.ctx.run_dir}", flush=True)
        try:
            results = base_results(job.ctx)
            stage_timings: dict[str, float] = {}
            prepare_bc_run_outputs(job.ctx.run_cfg, resume=job.ctx.resume)
            started = time.monotonic()
            results["bc"] = run_behavior_cloning(
                job.config,
                seed_checkpoint=job.seed_checkpoint,
                model_factory=job.model_factory,
                graph=job.graph,
                side_channel_provider=job.side_channel_provider,
                source_bundle=source_bundle,
                release_sources=False,
                log_label=_job_log_label(job),
                cancel_event=job.cancel_event,
            )
            stage_timings["bc"] = time.monotonic() - started
            results["stage_timings"] = stage_timings
            finalize_results(job.ctx, results, stage_timings)
            job.error = None
            done_list = self._done
            status = "done"
        except Exception:
            if job.cancel_event.is_set():
                job.error = "cancelled"
                status = "cancelled"
            else:
                job.error = traceback.format_exc(limit=16)
                status = "failed"
            done_list = self._failed
        finally:
            job.finished_at = time.time()
            with self._cv:
                self._active.pop(job.job_id, None)
                done_list.append(job)
                self._cv.notify_all()
            print(
                f"  [bc/daemon] {status} {job.job_id}: {job.ctx.run_dir}",
                flush=True,
            )
            if status == "failed" and job.error:
                # Surface the run-failure traceback to logs, not just the
                # one-word status — same loud-failure rationale as the
                # source-load path above.
                print(
                    f"  [bc/daemon] ERROR: {job.job_id} traceback:\n{job.error}",
                    file=sys.stderr,
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resident BC ablation daemon")
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Unix socket path")
    parser.add_argument("--parallel-runs", type=int, default=1)
    args = parser.parse_args()

    daemon = AblationDaemon(socket_path=Path(args.socket), parallel_runs=args.parallel_runs)

    def _stop(_signum: int, _frame: Any) -> None:
        daemon.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    daemon.start()
    daemon.serve_forever()


if __name__ == "__main__":
    main()
