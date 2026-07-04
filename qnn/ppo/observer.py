"""AlgoObserver that maintains the run's single best checkpoint.

SF's ``Learner.save_best`` keeps only the single most recent best checkpoint
(``keep=1``) under the SF experiment dir.  This observer connects to each
learner's ``saved_model`` signal and mirrors the newest best into
``{train_dir}/best/best_<run_id>.pth`` (overwritten in place — one best per
run, matching BC's ``best_<run_id>.pth``) and to the NAS share over SMB.

NAS connection details come from the ``QNN_NAS_*`` env vars (same contract
as corpus/nas.py; defaults nas.local/QNN/guest).

Usage::

    from qnn.ppo.observer import BestCheckpointArchiver
    runner.register_observer(BestCheckpointArchiver(runner))
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sample_factory.algo.runners.runner import Runner

from sample_factory.algo.runners.runner import AlgoObserver
from signal_slot.signal_slot import EventLoopObject

from qnn.utils.artifacts import best_name, new_run_id

log = logging.getLogger(__name__)


class BestCheckpointArchiver(AlgoObserver, EventLoopObject):
    """Mirror the newest best checkpoint to a run-id-named archive file."""

    def __init__(self, runner: "Runner") -> None:
        EventLoopObject.__init__(self, runner.event_loop, "BestCheckpointArchiver")
        self._last_source: str = ""
        self._archive_dir: Path | None = None
        self._exp_dir: Path | None = None
        self._run_id = ""
        self._smb_available = False
        self._nas_dir = ""

    def on_start(self, runner: "Runner") -> None:
        cfg = runner.cfg
        train_dir = Path(str(getattr(cfg, "train_dir", ".")))
        experiment = str(getattr(cfg, "experiment", "quake_combat"))
        self._run_id = str(getattr(cfg, "qnn_run_id", "") or "") or new_run_id()
        self._archive_dir = train_dir / "best"
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._exp_dir = train_dir / experiment

        # Try to register the SMB share for direct writes (no mount needed).
        try:
            import smbclient
            server = os.environ.get("QNN_NAS_SERVER", "nas.local")
            share = os.environ.get("QNN_NAS_SHARE", "QNN")
            user = os.environ.get("QNN_NAS_USER", "guest")
            password = os.environ.get("QNN_NAS_PASS", "guest")
            smbclient.ClientConfig(require_secure_negotiate=False)
            smbclient.register_session(
                server, username=user, password=password,
                encrypt=False, require_signing=False, auth_protocol="ntlm",
            )
            self._nas_dir = rf"\\{server}\{share}\best"
            smbclient.makedirs(self._nas_dir, exist_ok=True)
            self._smb_available = True
            log.info("SMB %s available for best checkpoint sync", self._nas_dir)
        except Exception:
            self._smb_available = False
            log.info("NAS not available — skipping best checkpoint sync")

    def on_connect_components(self, runner: "Runner") -> None:
        for learner_worker in runner.learners.values():
            learner_worker.saved_model.connect(self._on_saved_model)

    def _on_saved_model(self, policy_id: int) -> None:
        if self._archive_dir is None or self._exp_dir is None:
            return
        ckpt_dir = self._exp_dir / f"checkpoint_p{policy_id}"
        if not ckpt_dir.is_dir():
            return
        best_files = sorted(ckpt_dir.glob("best_0*.pth"))
        if not best_files:
            return
        newest = best_files[-1]
        if newest.name == self._last_source:
            return
        self._last_source = newest.name
        dest = self._archive_dir / best_name(self._run_id)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copy2(newest, tmp)
        os.replace(tmp, dest)
        log.info("Archived best checkpoint %s -> %s", newest.name, dest.name)
        if self._smb_available:
            self._smb_copy(dest, source_name=newest.name)

    def _smb_copy(self, src: Path, *, source_name: str) -> None:
        try:
            import smbclient
            nas_dest = self._nas_dir + "\\" + src.name
            with open(src, "rb") as local_f:
                with smbclient.open_file(nas_dest, mode="wb") as remote_f:
                    shutil.copyfileobj(local_f, remote_f)
            log.info("Copied %s (%s) to NAS", src.name, source_name)
        except Exception as exc:
            log.warning("Failed to copy %s to NAS: %s", src.name, exc)
