"""Fetch, merge, and compile a FrikBotNex mod gamedir."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GitSource:
    url: str
    commit: str


FRIKBOTNEX_SOURCE = GitSource(
    url="https://github.com/0xBrsm/FrikBotNex.git",
    commit="c767dbf325950b62a13ba22c68a395b39d6ad376",
)
QUAKE_TOOLS_SOURCE = GitSource(
    url="https://github.com/id-Software/Quake-Tools.git",
    commit="c0d1b91c74eb654365ac7755bc837e497caaca73",
)
FRIKBOTNEX_GAMEDIR = "frikbotnex"

_PROGS_INSERT = [
    "waypoints/map_dm1.qc",
    "waypoints/map_dm2.qc",
    "waypoints/map_dm3.qc",
    "waypoints/map_dm4.qc",
    "waypoints/map_dm5.qc",
    "waypoints/map_dm6.qc",
    "frikbot/bot.qc",
    "frikbot/bot_way.qc",
    "frikbot/bot_fight.qc",
    "frikbot/bot_ai.qc",
    "frikbot/bot_misc.qc",
    "frikbot/bot_phys.qc",
    "frikbot/bot_move.qc",
    "frikbot/bot_ed.qc",
]
_DEFS_OVERRIDES = [
    "void(entity e, float chan, string samp, float vol, float atten) sound = #8;",
    "void(entity client, string s)stuffcmd = #21;",
    "void(entity client, string s) sprint = #24;",
    "vector(entity e, float speed) aim = #44;\t\t// returns the shooting vector",
    "void(float to, float f) WriteByte\t\t= #52;",
    "void(float to, float f) WriteChar\t\t= #53;",
    "void(float to, float f) WriteShort\t\t= #54;",
    "void(float to, float f) WriteLong\t\t= #55;",
    "void(float to, float f) WriteCoord\t\t= #56;",
    "void(float to, float f) WriteAngle\t\t= #57;",
    "void(float to, string s) WriteString\t= #58;",
    "void(float to, entity s) WriteEntity\t= #59;",
    "void(entity client, string s) centerprint = #73;\t// sprint, but in middle",
    "void(entity e) setspawnparms\t\t= #78;\t\t// set parm1... to the",
]
_WORLDSPAWN_NEEDLE = "void() worldspawn =\n{\n\tlastspawn = world;\n\tInitBodyQue ();\n"
_WORLDSPAWN_REPLACEMENT = "void() worldspawn =\n{\n\tBotInit();\t// FrikBot\n\tlastspawn = world;\n\tInitBodyQue ();\n"
_STARTFRAME_NEEDLE = 'void() StartFrame =\n{\n\tteamplay = cvar("teamplay");\n'
_STARTFRAME_REPLACEMENT = 'void() StartFrame =\n{\n\tBotFrame();\t// FrikBot\n\tteamplay = cvar("teamplay");\n'
_PLAYERPRETHINK_NEEDLE = "void() PlayerPreThink =\n{\n"
_PLAYERPRETHINK_REPLACEMENT = "void() PlayerPreThink =\n{\n\tif (BotPreFrame())\n\t\treturn;\n"
_PLAYERPOSTTHINK_NEEDLE = "void() PlayerPostThink =\n{\n"
_PLAYERPOSTTHINK_REPLACEMENT = "void() PlayerPostThink =\n{\n\tif (BotPostFrame())\n\t\treturn;\n"
_CLIENTCONNECT_NEEDLE = "void() ClientConnect =\n{\n"
_CLIENTCONNECT_REPLACEMENT = "void() ClientConnect =\n{\n\tClientInRankings();\t// FrikBot\n"
_CLIENTDISCONNECT_NEEDLE = "void() ClientDisconnect =\n{\n"
_CLIENTDISCONNECT_REPLACEMENT = "void() ClientDisconnect =\n{\n\tClientDisconnected();\t// FrikBot\n"


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def _looks_like_quake_basedir(path: Path) -> bool:
    id1_dir = path / "id1"
    if not id1_dir.is_dir():
        return False
    pak_names = ("PAK0.PAK", "PAK1.PAK", "pak0.pak", "pak1.pak")
    return any((id1_dir / name).exists() for name in pak_names)


def _replace_once(text: str, old: str, new: str, *, path: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not patch {path}: missing expected text: {old!r}")
    return text.replace(old, new, 1)


def _merge_progs_src(text: str) -> str:
    lines = text.splitlines()
    merged: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.strip() == "../progs.dat":
            merged.append("progs.dat")
            continue
        merged.append(line)
        if not inserted and line.strip() == "defs.qc":
            merged.extend(_PROGS_INSERT)
            inserted = True
    if not inserted:
        raise RuntimeError("Could not patch progs.src: defs.qc entry not found")
    return "\n".join(merged) + "\n"


def _patch_defs_qc(text: str) -> str:
    patched = text
    for line in _DEFS_OVERRIDES:
        patched = _replace_once(patched, line, f"// FrikBot override: {line}", path="defs.qc")
    return patched


def _patch_world_qc(text: str) -> str:
    patched = _replace_once(text, _WORLDSPAWN_NEEDLE, _WORLDSPAWN_REPLACEMENT, path="world.qc")
    return _replace_once(patched, _STARTFRAME_NEEDLE, _STARTFRAME_REPLACEMENT, path="world.qc")


def _patch_client_qc(text: str) -> str:
    patched = _replace_once(text, _PLAYERPRETHINK_NEEDLE, _PLAYERPRETHINK_REPLACEMENT, path="client.qc")
    patched = _replace_once(patched, _PLAYERPOSTTHINK_NEEDLE, _PLAYERPOSTTHINK_REPLACEMENT, path="client.qc")
    patched = _replace_once(patched, _CLIENTCONNECT_NEEDLE, _CLIENTCONNECT_REPLACEMENT, path="client.qc")
    return _replace_once(patched, _CLIENTDISCONNECT_NEEDLE, _CLIENTDISCONNECT_REPLACEMENT, path="client.qc")


def prepare_frikbotnex_tree(target: Path, quakec_root: Path, frikbot_root: Path) -> Path:
    shutil.copytree(quakec_root, target, dirs_exist_ok=True)
    shutil.copytree(frikbot_root / "src" / "frikbot", target / "frikbot", dirs_exist_ok=True)
    shutil.copytree(frikbot_root / "src" / "waypoints", target / "waypoints", dirs_exist_ok=True)

    (target / "progs.src").write_text(_merge_progs_src((target / "progs.src").read_text(encoding="utf-8")), encoding="utf-8")
    (target / "defs.qc").write_text(_patch_defs_qc((target / "defs.qc").read_text(encoding="utf-8")), encoding="utf-8")
    (target / "world.qc").write_text(_patch_world_qc((target / "world.qc").read_text(encoding="utf-8")), encoding="utf-8")
    (target / "client.qc").write_text(_patch_client_qc((target / "client.qc").read_text(encoding="utf-8")), encoding="utf-8")
    return target


def _fetch_git_source(root: Path, source: GitSource, name: str) -> Path:
    checkout = root / name
    _run(["git", "init", "-q", str(checkout)])
    _run(["git", "-C", str(checkout), "remote", "add", "origin", source.url])
    _run(["git", "-C", str(checkout), "fetch", "-q", "--depth", "1", "origin", source.commit])
    _run(["git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD"])
    return checkout


def _resolve_compiler(explicit: str | None) -> str:
    candidate = explicit or shutil.which("fteqcc")
    if candidate:
        return candidate
    raise RuntimeError(
        "fteqcc is required to build FrikBotNex. Install it in the devcontainer or use src/scripts/train-container.sh install-frikbotnex."
    )


def _compile_frikbotnex(build_root: Path, compiler: str) -> Path:
    _run([compiler, "-Fautoproto", "-O0"], cwd=build_root)
    output = build_root / "progs.dat"
    if not output.exists():
        raise RuntimeError(f"fteqcc completed without writing {output}")
    return output


def _write_notice(target_dir: Path) -> None:
    notice = (
        "FrikBotNex build output\n"
        "\n"
        "This gamedir was generated from:\n"
        f"- {FRIKBOTNEX_SOURCE.url} @ {FRIKBOTNEX_SOURCE.commit}\n"
        f"- {QUAKE_TOOLS_SOURCE.url} @ {QUAKE_TOOLS_SOURCE.commit}\n"
        "\n"
        "The FrikBot source header states that the bot code is in the public domain.\n"
        "The Quake v1.01 QuakeC base comes from id Software's Quake-Tools repository and remains under GPLv2-or-later.\n"
    )
    (target_dir / "NOTICE.txt").write_text(notice, encoding="utf-8")


def install_frikbotnex(asset_root: Path, *, compiler: str | None = None, force: bool = False) -> dict[str, object]:
    asset_root = asset_root.resolve()
    if not _looks_like_quake_basedir(asset_root):
        raise RuntimeError(f"{asset_root} is not a Quake basedir; expected {asset_root / 'id1'} with pak files")

    compiler_path = _resolve_compiler(compiler)
    target_dir = asset_root / FRIKBOTNEX_GAMEDIR
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "install_manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("frikbotnex_commit") == FRIKBOTNEX_SOURCE.commit
            and existing.get("quake_tools_commit") == QUAKE_TOOLS_SOURCE.commit
            and (target_dir / "progs.dat").exists()
        ):
            return existing

    with tempfile.TemporaryDirectory(prefix="frikbotnex-install.") as tmp_dir:
        temp_root = Path(tmp_dir)
        sources_root = temp_root / "sources"
        sources_root.mkdir()

        frikbot_checkout = _fetch_git_source(sources_root, FRIKBOTNEX_SOURCE, "FrikBotNex")
        quake_tools_checkout = _fetch_git_source(sources_root, QUAKE_TOOLS_SOURCE, "Quake-Tools")

        build_root = temp_root / "build"
        prepare_frikbotnex_tree(build_root, quake_tools_checkout / "qcc" / "v101qc", frikbot_checkout)
        output = _compile_frikbotnex(build_root, compiler_path)

        shutil.copy2(output, target_dir / "progs.dat")
        copying = quake_tools_checkout / "qcc" / "COPYING"
        if copying.exists():
            shutil.copy2(copying, target_dir / "COPYING.id-software")
        _write_notice(target_dir)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_root": str(asset_root),
        "gamedir": FRIKBOTNEX_GAMEDIR,
        "compiler": compiler_path,
        "frikbotnex_url": FRIKBOTNEX_SOURCE.url,
        "frikbotnex_commit": FRIKBOTNEX_SOURCE.commit,
        "quake_tools_url": QUAKE_TOOLS_SOURCE.url,
        "quake_tools_commit": QUAKE_TOOLS_SOURCE.commit,
        "outputs": [str(target_dir / "progs.dat"), str(target_dir / "NOTICE.txt")],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a pinned FrikBotNex gamedir under a Quake asset root")
    parser.add_argument("--asset-root", required=True, help="Quake basedir containing id1/")
    parser.add_argument("--compiler", default=None, help="Override the fteqcc executable path")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the installed manifest already matches the pinned commits")
    args = parser.parse_args()

    manifest = install_frikbotnex(Path(args.asset_root), compiler=args.compiler, force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
