from __future__ import annotations

import json
from pathlib import Path

import quake_ai.live_training as live_training
from quake_ai.live_training import (
    LiveProfile,
    PROFILES,
    RuntimePlan,
    _all_action_requires_collect,
    _required_gamedir,
    _resolve_asset_root,
    _resolve_demo_dir,
    _runtime_plan,
    _validate_native_mod_assets,
    run_live_pipeline,
)
from quake_ai.utils.io import read_json, write_json


def test_runtime_plan_scales_corpus_profile() -> None:
    runtime = {
        "requested_device": "gpu",
        "resolved_device": "cuda",
        "backend": "rocm",
        "cpu_count": 32,
        "cpu_affinity_count": 32,
        "devices": [{"index": 0, "name": "AMD GPU", "total_memory": 24 * 1024**3}],
    }

    plan = _runtime_plan(PROFILES["corpus"], runtime)

    assert plan.num_envs == 30
    assert plan.rollout_steps == 256
    assert plan.total_steps >= plan.num_envs * plan.rollout_steps * 64
    assert plan.bc_batch_size == 8192
    assert plan.minibatch_size == 2048
    assert plan.eval_episodes == 32


def test_runtime_plan_caps_verify_profile() -> None:
    runtime = {
        "requested_device": "gpu",
        "resolved_device": "cuda",
        "backend": "rocm",
        "cpu_count": 64,
        "cpu_affinity_count": 64,
        "devices": [{"index": 0, "name": "AMD GPU", "total_memory": 8 * 1024**3}],
    }

    plan = _runtime_plan(PROFILES["verify"], runtime)

    assert plan.num_envs == 8
    assert plan.rollout_steps == 64
    assert plan.bc_batch_size == 1024
    assert plan.minibatch_size == 256


def test_bot_ladder_profiles_are_registered_with_notes() -> None:
    ladder_profiles = {name: profile for name, profile in PROFILES.items() if profile.profile_group == "combat_bot_ladder"}

    assert {"combat-bot-duel-dm2-verify", "combat-bot-open-dm4", "combat-bot-vertical-dm3-verify", "combat-bot-pressure-dm6"}.issubset(
        ladder_profiles
    )
    assert ladder_profiles["combat-bot-open-dm4"].retained_role == "main_retained"
    assert "aim" in ladder_profiles["combat-bot-duel-dm2-verify"].profile_note.lower()


def test_resolve_asset_root_from_environment(monkeypatch, tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    id1_dir = asset_root / "id1"
    id1_dir.mkdir(parents=True)
    (id1_dir / "PAK0.PAK").write_bytes(b"pak")
    monkeypatch.setenv("QUAKE_BASEDIR", str(asset_root))

    assert _resolve_asset_root(None) == asset_root


def test_resolve_demo_dir_accepts_uppercase_dem(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demos"
    demo_dir.mkdir()
    (demo_dir / "MATCH01.DEM").write_bytes(b"demo")

    profile = LiveProfile(
        name="uppercase-demo",
        bc_config="unused",
        ppo_config="unused",
        eval_config="unused",
        default_demo_dir=str(demo_dir),
        auto_demo_dirs=(str(demo_dir),),
        collect_out=str(tmp_path / "collect"),
        plan_path=str(tmp_path / "plan.json"),
    )

    assert _resolve_demo_dir(profile, None) == demo_dir


def test_required_gamedir_extracts_game_argument() -> None:
    assert _required_gamedir(["-listen", "-game", "frikbotnex"]) == "frikbotnex"
    assert _required_gamedir(["-game", ""]) is None
    assert _required_gamedir(["-listen"]) is None


def test_validate_native_mod_assets_requires_compiled_gamedir(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    id1_dir = asset_root / "id1"
    id1_dir.mkdir(parents=True)
    (id1_dir / "PAK0.PAK").write_bytes(b"pak")

    try:
        _validate_native_mod_assets(asset_root, ["-game", "frikbotnex"])
    except RuntimeError as exc:
        assert "install-frikbotnex" in str(exc)
    else:
        raise AssertionError("expected missing gamedir to raise")

    mod_dir = asset_root / "frikbotnex"
    mod_dir.mkdir()
    (mod_dir / "progs.dat").write_bytes(b"compiled")
    _validate_native_mod_assets(asset_root, ["-game", "frikbotnex"])


def test_all_action_requires_collect_only_when_downstream_uses_collect_dir(tmp_path: Path) -> None:
    collect_dir = tmp_path / "collect"
    profile = LiveProfile(
        name="collect-check",
        bc_config="unused",
        ppo_config="unused",
        eval_config="unused",
        default_demo_dir="unused",
        auto_demo_dirs=("unused",),
        collect_out=str(collect_dir),
        plan_path=str(tmp_path / "plan.json"),
    )

    assert _all_action_requires_collect(
        profile,
        {"world_ticks_path": str(collect_dir / "world_ticks.ndjson")},
        {"map_features_path": str(tmp_path / "elsewhere" / "map_features.json")},
    )
    assert not _all_action_requires_collect(
        profile,
        {"world_ticks_path": str(tmp_path / "retained" / "world_ticks.ndjson")},
        {"map_features_path": str(tmp_path / "elsewhere" / "map_features.json")},
    )


def test_all_action_skips_redundant_collect(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "skip_collect"
    root.mkdir()
    demo_dir = root / "demos"
    demo_dir.mkdir()
    asset_root = root / "assets"
    asset_root.mkdir()
    worker_path = root / "worker"
    worker_path.write_bytes(b"worker")

    bc_cfg_path = root / "bc.json"
    ppo_cfg_path = root / "ppo.json"
    eval_cfg_path = root / "eval.json"
    write_json(
        bc_cfg_path,
        {
            "map_id": "dm6",
            "world_ticks_path": str(root / "retained" / "world_ticks.ndjson"),
            "map_state_path": str(root / "retained" / "world_map.json"),
            "observation_format": "world_v2_competitive",
            "output_dir": str(root / "bc"),
        },
    )
    write_json(
        ppo_cfg_path,
        {
            "map_features_path": str(root / "retained" / "map_features.json"),
            "output_dir": str(root / "ppo"),
            "init_ckpt": str(root / "retained" / "bc_best_model.npz"),
            "observation_format": "world_v2_competitive",
        },
    )
    write_json(
        eval_cfg_path,
        {
            "map_features_path": str(root / "retained" / "map_features.json"),
            "checkpoint_path": str(root / "ppo" / "ppo_model.npz"),
            "output_dir": str(root / "eval"),
            "observation_format": "world_v2_competitive",
        },
    )

    profile = LiveProfile(
        name="skip-collect",
        bc_config=str(bc_cfg_path),
        ppo_config=str(ppo_cfg_path),
        eval_config=str(eval_cfg_path),
        default_demo_dir=str(demo_dir),
        auto_demo_dirs=(str(demo_dir),),
        collect_out=str(root / "collect"),
        plan_path=str(root / "live_training_plan.json"),
    )
    monkeypatch.setitem(PROFILES, "skip-collect", profile)

    runtime = {
        "requested_device": "gpu",
        "resolved_device": "cuda",
        "backend": "rocm",
        "cpu_count": 8,
        "cpu_affinity_count": 8,
        "devices": [{"index": 0, "name": "AMD GPU", "total_memory": 8 * 1024**3}],
    }
    plan = RuntimePlan(
        requested_device="gpu",
        resolved_device="cuda",
        backend="rocm",
        cpu_count=8,
        cpu_affinity_count=8,
        gpu_memory_bytes=8 * 1024**3,
        bc_batch_size=512,
        num_envs=4,
        rollout_steps=64,
        total_steps=4096,
        minibatch_size=256,
        eval_episodes=8,
    )
    monkeypatch.setattr(live_training, "build_runtime_plan", lambda profile_name, requested_device: (profile, runtime, plan))
    monkeypatch.setattr(live_training, "_resolve_demo_dir", lambda profile, explicit: demo_dir)
    monkeypatch.setattr(live_training, "_resolve_asset_root", lambda explicit: asset_root)
    monkeypatch.setattr(live_training, "_ensure_worker", lambda path, rebuild: worker_path)

    calls: list[str] = []

    def _unexpected_collect(**kwargs):
        raise AssertionError("collect_from_demos should be skipped when downstream stages use retained inputs")

    def _fake_bc(cfg):
        calls.append("bc")
        return {"test_accuracy": 0.8}

    def _fake_eval(cfg):
        calls.append("eval")
        return {"completion_rate": 0.0}

    def _fake_ppo(cfg):
        calls.append("ppo")
        return {"completion_rate": 0.0}

    def _fake_report(**kwargs):
        target = root / "report.json"
        target.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {
            "report_path": str(target),
            "operational_note_json_path": str(target),
            "operational_note_md_path": str(target),
        }

    monkeypatch.setattr(live_training, "collect_from_demos", _unexpected_collect)
    monkeypatch.setattr(live_training, "run_behavior_cloning", _fake_bc)
    monkeypatch.setattr(live_training, "run_evaluation", _fake_eval)
    monkeypatch.setattr(live_training, "run_ppo", _fake_ppo)
    monkeypatch.setattr(live_training, "_write_run_report", _fake_report)

    result = run_live_pipeline(
        profile_name="skip-collect",
        action="all",
        device="gpu",
        demo_dir=None,
        asset_root=None,
        worker_binary=str(worker_path),
        rebuild_worker=False,
    )

    assert result["collect"]["skipped"] is True
    assert "retained inputs" in result["collect"]["reason"]
    assert calls == ["bc", "eval", "ppo", "eval"]


def test_all_action_switches_ppo_init_ckpt_to_local_bc_on_obs_dim_mismatch(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ppo_init_switch"
    root.mkdir()
    demo_dir = root / "demos"
    demo_dir.mkdir()
    (demo_dir / "sample.dem").write_text("demo", encoding="utf-8")
    asset_root = root / "assets"
    asset_root.mkdir()
    worker_path = root / "worker"
    worker_path.write_bytes(b"worker")

    retained_dir = root / "retained"
    retained_dir.mkdir()
    legacy_ckpt = retained_dir / "bc_best_model.npz"
    legacy_ckpt.write_bytes(b"legacy")
    write_json(retained_dir / "bc_best_model.json", {"obs_dim": 126})

    bc_cfg_path = root / "bc.json"
    ppo_cfg_path = root / "ppo.json"
    eval_cfg_path = root / "eval.json"
    bc_dir = root / "bc"
    ppo_dir = root / "ppo"
    eval_dir = root / "eval"
    write_json(
        bc_cfg_path,
        {
            "map_id": "dm4",
            "world_ticks_path": str(root / "retained" / "world_ticks.ndjson"),
            "map_state_path": str(root / "retained" / "world_map.json"),
            "observation_format": "world_v2_competitive",
            "output_dir": str(bc_dir),
        },
    )
    write_json(
        ppo_cfg_path,
        {
            "map_features_path": str(root / "retained" / "map_features.json"),
            "output_dir": str(ppo_dir),
            "init_ckpt": str(legacy_ckpt),
            "observation_format": "world_v2_competitive",
        },
    )
    write_json(
        eval_cfg_path,
        {
            "map_features_path": str(root / "retained" / "map_features.json"),
            "checkpoint_path": str(ppo_dir / "ppo_model.npz"),
            "output_dir": str(eval_dir),
            "observation_format": "world_v2_competitive",
        },
    )

    profile = LiveProfile(
        name="ppo-init-switch",
        bc_config=str(bc_cfg_path),
        ppo_config=str(ppo_cfg_path),
        eval_config=str(eval_cfg_path),
        default_demo_dir=str(demo_dir),
        auto_demo_dirs=(str(demo_dir),),
        collect_out=str(root / "collect"),
        plan_path=str(root / "live_training_plan.json"),
    )
    monkeypatch.setitem(PROFILES, "ppo-init-switch", profile)

    runtime = {
        "requested_device": "gpu",
        "resolved_device": "cuda",
        "backend": "rocm",
        "cpu_count": 8,
        "cpu_affinity_count": 8,
        "devices": [{"index": 0, "name": "AMD GPU", "total_memory": 8 * 1024**3}],
    }
    plan = RuntimePlan(
        requested_device="gpu",
        resolved_device="cuda",
        backend="rocm",
        cpu_count=8,
        cpu_affinity_count=8,
        gpu_memory_bytes=8 * 1024**3,
        bc_batch_size=512,
        num_envs=4,
        rollout_steps=64,
        total_steps=4096,
        minibatch_size=256,
        eval_episodes=8,
    )
    monkeypatch.setattr(live_training, "build_runtime_plan", lambda profile_name, requested_device: (profile, runtime, plan))
    monkeypatch.setattr(live_training, "_resolve_demo_dir", lambda profile, explicit: demo_dir)
    monkeypatch.setattr(live_training, "_resolve_asset_root", lambda explicit: asset_root)
    monkeypatch.setattr(live_training, "_ensure_worker", lambda path, rebuild: worker_path)

    captured: dict[str, str] = {}

    def _fake_bc(cfg):
        bc_dir.mkdir(parents=True, exist_ok=True)
        (bc_dir / "bc_best_model.npz").write_bytes(b"current")
        write_json(bc_dir / "bc_best_model.json", {"obs_dim": 234})
        return {"test_accuracy": 0.8}

    def _fake_eval(cfg):
        return {"completion_rate": 0.0}

    def _fake_ppo(cfg):
        captured["init_ckpt"] = cfg.init_ckpt
        return {"completion_rate": 0.0}

    def _fake_report(**kwargs):
        target = root / "report.json"
        target.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {
            "report_path": str(target),
            "operational_note_json_path": str(target),
            "operational_note_md_path": str(target),
        }

    monkeypatch.setattr(live_training, "run_behavior_cloning", _fake_bc)
    monkeypatch.setattr(live_training, "run_evaluation", _fake_eval)
    monkeypatch.setattr(live_training, "run_ppo", _fake_ppo)
    monkeypatch.setattr(live_training, "_write_run_report", _fake_report)

    result = run_live_pipeline(
        profile_name="ppo-init-switch",
        action="all",
        device="gpu",
        demo_dir=None,
        asset_root=None,
        worker_binary=str(worker_path),
        rebuild_worker=False,
    )

    assert captured["init_ckpt"] == str(bc_dir / "bc_best_model.npz")
    assert result["ppo_init_ckpt"] == str(bc_dir / "bc_best_model.npz")
    assert "obs_dim=126" in result["ppo_init_ckpt_note"]
    assert "obs_dim=234" in result["ppo_init_ckpt_note"]


def test_plan_action_does_not_require_installed_mod(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "plan_profile"
    demo_dir = root / "demos"
    demo_dir.mkdir(parents=True)
    (demo_dir / "sample.dem").write_text("demo", encoding="utf-8")
    asset_root = root / "assets"
    asset_root.mkdir()

    profile = LiveProfile(
        name="combat-bot-plan-test",
        bc_config="configs/bc_combat_bot_verify.yaml",
        ppo_config="configs/ppo_combat_bot_verify.yaml",
        eval_config="configs/eval_combat_bot_verify.yaml",
        default_demo_dir=str(demo_dir),
        auto_demo_dirs=(str(demo_dir),),
        collect_out=str(root / "collect"),
        plan_path=str(root / "live_training_plan.json"),
        ppo_overrides={"native_args": ["-game", "frikbotnex"]},
        eval_overrides={"native_args": ["-game", "frikbotnex"]},
    )
    monkeypatch.setitem(PROFILES, "combat-bot-plan-test", profile)

    runtime = {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "backend": "cpu",
        "cpu_count": 4,
        "cpu_affinity_count": 4,
        "devices": [],
    }
    plan = RuntimePlan(
        requested_device="cpu",
        resolved_device="cpu",
        backend="cpu",
        cpu_count=4,
        cpu_affinity_count=4,
        gpu_memory_bytes=0,
        bc_batch_size=64,
        num_envs=2,
        rollout_steps=32,
        total_steps=512,
        minibatch_size=32,
        eval_episodes=4,
    )

    monkeypatch.setattr(live_training, "build_runtime_plan", lambda profile_name, requested_device: (profile, runtime, plan))
    monkeypatch.setattr(live_training, "_resolve_demo_dir", lambda profile, explicit: demo_dir)

    result = run_live_pipeline(
        profile_name="combat-bot-plan-test",
        action="plan",
        device="cpu",
        demo_dir=None,
        asset_root=str(asset_root),
        worker_binary=str(root / "worker"),
        rebuild_worker=False,
    )

    assert result["profile"] == "combat-bot-plan-test"
    assert Path(result["plan_path"]).exists()


def test_report_action_writes_operational_note(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "report_profile"
    collect_dir = root / "collect"
    bc_dir = root / "bc"
    eval_bc_dir = root / "eval_bc"
    ppo_dir = root / "ppo"
    eval_dir = root / "eval"
    for directory in (collect_dir, bc_dir, eval_bc_dir, ppo_dir, eval_dir):
        directory.mkdir(parents=True)

    write_json(collect_dir / "collect_manifest.json", {"world_ticks": "collect/world_ticks.ndjson"})
    write_json(
        root / "live_training_plan.json",
        {
            "runtime": {
                "backend": "rocm",
                "resolved_device": "cuda",
                "devices": [{"name": "AMD Test GPU", "total_memory": 12 * 1024**3}],
            },
            "plan": {
                "bc_batch_size": 512,
                "eval_episodes": 8,
                "num_envs": 4,
                "rollout_steps": 32,
            },
        },
    )
    write_json(
        bc_dir / "bc_manifest.json",
        {"metrics": {"test_accuracy": 0.91}},
    )
    write_json(
        bc_dir / "bc_summary.json",
        {"test_accuracy": 0.91},
    )
    (bc_dir / "bc_best_model.npz").write_bytes(b"bc")
    write_json(
        eval_bc_dir / "eval_manifest.json",
        {
            "metrics": {
                "modes": {
                    "greedy": {"completion_rate": 0.375, "stuck_rate": 0.5},
                    "sampled": {"completion_rate": 0.5, "stuck_rate": 0.25},
                }
            }
        },
    )
    write_json(eval_bc_dir / "eval_summary.json", {"status": "ok"})
    write_json(eval_bc_dir / "model_card.json", {"model": {"checkpoint": "bc/bc_best_model.npz"}})
    write_json(
        ppo_dir / "ppo_manifest.json",
        {"metrics": {"completion_rate": 0.25, "episodes_completed": 12, "steps_done": 512}},
    )
    write_json(
        ppo_dir / "ppo_summary.json",
        {"completion_rate": 0.25, "episodes_completed": 12, "steps_done": 512},
    )
    (ppo_dir / "ppo_model.npz").write_bytes(b"ppo")
    write_json(
        eval_dir / "eval_manifest.json",
        {
            "metrics": {
                "modes": {
                    "greedy": {"completion_rate": 0.5, "stuck_rate": 0.25},
                    "sampled": {"completion_rate": 0.375, "stuck_rate": 0.125},
                }
            }
        },
    )
    write_json(eval_dir / "eval_summary.json", {"status": "ok"})
    write_json(eval_dir / "model_card.json", {"model": {"checkpoint": "ppo/ppo_model.npz"}})

    profile = LiveProfile(
        name="report-test",
        bc_config="unused",
        ppo_config="unused",
        eval_config="unused",
        default_demo_dir="unused",
        auto_demo_dirs=("unused",),
        collect_out=str(collect_dir),
        plan_path=str(root / "live_training_plan.json"),
    )
    monkeypatch.setitem(PROFILES, "report-test", profile)

    result = run_live_pipeline(
        profile_name="report-test",
        action="report",
        device="cpu",
        demo_dir=None,
        asset_root=None,
        worker_binary="unused",
        rebuild_worker=False,
    )

    note_path = Path(result["operational_note_json_path"])
    md_path = Path(result["operational_note_md_path"])
    report_path = Path(result["report_path"])

    assert note_path.exists()
    assert md_path.exists()
    assert report_path.exists()

    note = read_json(note_path)
    assert note["worker_count"] == 4
    assert note["metrics"]["eval_bc_greedy_completion_rate"] == 0.375
    assert note["metrics"]["ppo_completion_rate"] == 0.25
    assert note["metrics"]["eval_greedy_completion_rate"] == 0.5
    assert "AMD Test GPU" in note["device"]
    assert "PPO improved greedy completion over the BC checkpoint" in md_path.read_text(encoding="utf-8")
    assert "Live Training Operational Note" in md_path.read_text(encoding="utf-8")
