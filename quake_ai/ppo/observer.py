"""AlgoObserver that archives every best checkpoint to a profile-level directory.

SF's ``Learner.save_best`` keeps only the single most recent best checkpoint
(``keep=1``).  This observer connects to each learner's ``saved_model`` signal
and archives new best checkpoints into ``{train_dir}/best/`` (hard-link or
copy) and to the NAS share over SMB.

Usage::

    from quake_ai.ppo.observer import BestCheckpointArchiver
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

log = logging.getLogger(__name__)

_NAS_SHARE = r"\\pi.local\nqcorpus"
_NAS_BEST = _NAS_SHARE + r"\best"


class BestCheckpointArchiver(AlgoObserver, EventLoopObject):
    """Archive best checkpoints whenever a model is saved."""

    def __init__(self, runner: "Runner") -> None:
        EventLoopObject.__init__(self, runner.event_loop, "BestCheckpointArchiver")
        self._seen: set[str] = set()
        self._archive_dir: Path | None = None
        self._exp_dir: Path | None = None
        self._smb_available = False

    def on_start(self, runner: "Runner") -> None:
        cfg = runner.cfg
        train_dir = Path(str(getattr(cfg, "train_dir", ".")))
        experiment = str(getattr(cfg, "experiment", "quake_combat"))
        self._archive_dir = train_dir / "best"
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._exp_dir = train_dir / experiment

        # Locate the demos directory so we can archive best.dem alongside
        # the best checkpoint.  Demos live in <basedir>/<game>/demos/.
        basedir = Path(str(getattr(cfg, "quake_basedir", "assets")))
        native_args = str(getattr(cfg, "quake_native_args_json", "") or "")
        game_subdir = "id1"
        if "-game" in native_args:
            import json
            try:
                args = json.loads(native_args)
                idx = args.index("-game")
                if idx + 1 < len(args):
                    game_subdir = args[idx + 1]
            except (json.JSONDecodeError, ValueError):
                pass
        self._demos_dir = basedir / game_subdir / "demos"

        # Seed _seen with any files already in the archive so we don't
        # re-copy them on resume.
        for f in self._archive_dir.glob("best_*.pth"):
            self._seen.add(f.name)

        # Try to register the SMB share for direct writes (no mount needed).
        try:
            import smbclient
            smbclient.ClientConfig(username="guest", password="", require_secure_negotiate=False)
            smbclient.register_session(
                "pi.local", username="guest", password="",
                auth_protocol="ntlm", require_signing=False,
            )
            smbclient.makedirs(_NAS_BEST, exist_ok=True)
            self._smb_available = True
            log.info("SMB share %s available for best checkpoint sync", _NAS_SHARE)
        except Exception:
            self._smb_available = False
            log.info("SMB share %s not available — skipping NAS sync", _NAS_SHARE)

    def on_connect_components(self, runner: "Runner") -> None:
        for learner_worker in runner.learners.values():
            learner_worker.saved_model.connect(self._on_saved_model)

    def _on_saved_model(self, policy_id: int) -> None:
        if self._archive_dir is None or self._exp_dir is None:
            return
        ckpt_dir = self._exp_dir / f"checkpoint_p{policy_id}"
        if not ckpt_dir.is_dir():
            return
        for best_file in ckpt_dir.glob("best_0*.pth"):
            if best_file.name in self._seen:
                continue
            self._seen.add(best_file.name)
            dest = self._archive_dir / best_file.name
            if dest.exists():
                continue
            try:
                os.link(best_file, dest)
            except OSError:
                shutil.copy2(best_file, dest)
            log.info("Archived best checkpoint: %s", best_file.name)
            self._archive_best_demo(best_file.name, policy_id)
            if self._smb_available:
                self._smb_copy(best_file)

    def _archive_best_demo(self, checkpoint_name: str, policy_id: int = 0) -> None:
        """Copy the most recent worker demo for *policy_id*, named to match the checkpoint.

        Layout:
            best/*.pth
            best/demos/*.dem
            best/demos/maps/*.bsp   (original map name, e.g. gen_1234567.bsp)
        """
        if not self._demos_dir or not self._demos_dir.is_dir():
            return
        # Find the most recently completed demo for this policy.
        # Envs save a copy of the finished episode as *_last.dem before reset
        # overwrites the active demo file.
        worker_demos = list(self._demos_dir.glob(f"train_p{policy_id}_w*_last.dem"))
        if not worker_demos:
            worker_demos = list(self._demos_dir.glob(f"train_p{policy_id}_w*.dem"))
            worker_demos = [d for d in worker_demos if "_last" not in d.name]
        if not worker_demos:
            return
        newest = max(worker_demos, key=lambda p: p.stat().st_mtime)
        # Name demo to match checkpoint: best_000002890_12231680_reward_-23.041.dem
        demo_stem = Path(checkpoint_name).stem  # strip .pth
        demos_dir = self._archive_dir / "demos"
        demos_dir.mkdir(exist_ok=True)
        dest_dem = demos_dir / f"{demo_stem}.dem"
        shutil.copy2(newest, dest_dem)
        # Copy the matching procgen BSP if it exists (read map name from demo).
        # Keep the original map name so it can be loaded for replay.
        try:
            import re
            raw = newest.read_bytes()
            m = re.search(rb"(gen_\d+)", raw[:4096])
            if m:
                map_id = m.group(1).decode()
                # Maps live under <game_dir>/maps/, same parent as demos dir.
                bsp_src = self._demos_dir.parent / "maps" / f"{map_id}.bsp"
                if bsp_src.exists():
                    maps_dir = demos_dir / "maps"
                    maps_dir.mkdir(exist_ok=True)
                    dest_bsp = maps_dir / f"{map_id}.bsp"
                    if not dest_bsp.exists():
                        shutil.copy2(bsp_src, dest_bsp)
        except Exception:
            pass
        log.info("Archived best demo: %s", dest_dem.name)

    def _smb_copy(self, src: Path) -> None:
        try:
            import smbclient
            nas_dest = _NAS_BEST + "\\" + src.name
            with open(src, "rb") as local_f:
                with smbclient.open_file(nas_dest, mode="wb") as remote_f:
                    shutil.copyfileobj(local_f, remote_f)
            log.info("Copied %s to NAS", src.name)
        except Exception as exc:
            log.warning("Failed to copy %s to NAS: %s", src.name, exc)
