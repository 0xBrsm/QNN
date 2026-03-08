"""Corpus manifest helpers and demo materialization."""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping
from urllib.parse import urlparse

try:
    import smbclient
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    smbclient = None

_CANONICAL_MAP_RE = re.compile(r"(?:^|[^a-z0-9])(e[1-4]m[1-8]|dm[1-6]|end)(?:[^a-z0-9]|$)")
_REGISTERED_REMOTE_SESSIONS: set[tuple[str, str, str]] = set()


def canonical_map_id(value: str) -> str | None:
    if not value:
        return None
    leaf = value.replace("\\", "/").split("/")[-1]
    stem = leaf.rsplit(".", 1)[0].lower() if "." in leaf else leaf.lower()
    prefix = stem.split("_", 1)[0]
    match = re.match(r"^(e[1-4]m[1-8]|dm[1-6]|end)", prefix)
    if match:
        return match.group(1)
    match = _CANONICAL_MAP_RE.search(stem)
    if match:
        return match.group(1)
    return None


def manifest_row_map_id(row: Mapping[str, object]) -> str | None:
    candidates = [
        str(row.get("extracted_dem_path", "")),
        str(row.get("local_path", "")),
    ]
    url = str(row.get("url", ""))
    if url:
        candidates.append(urlparse(url).path)
    for value in candidates:
        map_id = canonical_map_id(value)
        if map_id is not None:
            return map_id
    return None


def load_manifest_rows(path: str | Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(dict(json.loads(line)))
    return rows


def filter_manifest_rows(rows: Iterable[Mapping[str, object]], map_id: str) -> List[Dict[str, object]]:
    target = map_id.lower()
    filtered = []
    for row in rows:
        if manifest_row_map_id(row) == target:
            filtered.append(dict(row))
    return filtered


def _remote_join(share: str, relative_path: str) -> str:
    rel = relative_path.replace("/", "\\").strip("\\")
    if not rel:
        return share
    return share.rstrip("\\") + "\\" + rel


def _prefixed_relative_path(relative_path: str, remote_prefix: str = "") -> str:
    rel = relative_path.replace("\\", "/").strip("/")
    prefix = remote_prefix.replace("\\", "/").strip("/")
    if not prefix:
        return rel
    if not rel:
        return prefix
    return f"{prefix}/{rel}"


def _split_remote_storage_root(storage_root: str) -> tuple[str, str]:
    stripped = storage_root.replace("/", "\\").lstrip("\\")
    parts = [part for part in stripped.split("\\") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid remote storage root {storage_root!r}")
    host, share, *subdir = parts
    share_path = f"//{host}/{share}"
    base_dir = "/".join(subdir)
    return share_path, base_dir


def _quote_smbclient_arg(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _read_remote_bytes_with_cli(
    storage_root: str,
    relative_path: str,
    remote_username: str,
    remote_password: str,
) -> bytes:
    smbclient_bin = shutil.which("smbclient")
    if not smbclient_bin:
        raise RuntimeError("smbclient CLI is required to read remote corpus storage")

    share_path, base_dir = _split_remote_storage_root(storage_root)
    remote_path = _prefixed_relative_path(relative_path, base_dir)
    if not remote_path:
        raise ValueError("Missing remote path for SMB fetch")

    with tempfile.TemporaryDirectory(prefix="quake_ai_smb_") as temp_dir:
        target = Path(temp_dir) / "remote_payload.bin"
        command = f"get {_quote_smbclient_arg(remote_path)} {_quote_smbclient_arg(str(target))}"
        proc = subprocess.run(
            [smbclient_bin, share_path, "-U", f"{remote_username}%{remote_password}", "-c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "unknown smbclient error"
            raise RuntimeError(f"smbclient get failed for {storage_root}/{relative_path}: {detail}")
        return target.read_bytes()


def _register_remote_session(share: str, username: str, password: str) -> None:
    if smbclient is None:
        raise RuntimeError("smbclient is required to read remote corpus storage")
    host = share.lstrip("\\").split("\\", 1)[0]
    cache_key = (host, username, password)
    if cache_key in _REGISTERED_REMOTE_SESSIONS:
        return
    smbclient.ClientConfig(require_secure_negotiate=False)
    smbclient.register_session(
        host,
        username=username,
        password=password,
        encrypt=False,
        require_signing=False,
    )
    _REGISTERED_REMOTE_SESSIONS.add(cache_key)


def _read_storage_bytes(
    row: Mapping[str, object],
    relative_path: str,
    remote_username: str,
    remote_password: str,
    *,
    remote_prefix: str = "",
) -> bytes:
    backend = str(row.get("storage_backend", "local"))
    storage_root = str(row.get("storage_root", ""))
    if backend == "remote":
        remote_path = _prefixed_relative_path(relative_path, remote_prefix)
        if smbclient is not None:
            _register_remote_session(storage_root, remote_username, remote_password)
            with smbclient.open_file(_remote_join(storage_root, remote_path), mode="rb") as handle:
                return handle.read()
        return _read_remote_bytes_with_cli(storage_root, remote_path, remote_username, remote_password)
    return (Path(storage_root) / relative_path).read_bytes()


def _extract_with_dzip(payload: bytes, source_name: str) -> bytes | None:
    dzip_bin = os.environ.get("DZIP_BIN") or shutil.which("dzip")
    if not dzip_bin:
        return None

    with tempfile.TemporaryDirectory(prefix="quake_ai_dzip_") as temp_dir:
        source_path = Path(temp_dir) / (Path(source_name).name or "demo.dz")
        source_path.write_bytes(payload)
        proc = subprocess.run(
            [dzip_bin, "-x", "-f", str(source_path)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown dzip error"
            raise ValueError(f"dzip failed for {source_name}: {stderr}")

        extracted = sorted(
            path
            for path in Path(temp_dir).iterdir()
            if path.is_file() and path.suffix.lower() == ".dem" and path != source_path
        )
        if not extracted:
            raise ValueError(f"dzip extracted no demo from {source_name}")
        return extracted[0].read_bytes()


def decompress_demo_payload(payload: bytes, source_name: str = "") -> bytes:
    source = source_name.lower()
    if source.endswith(".dem"):
        return payload
    if payload.startswith(b"DZ"):
        extracted = _extract_with_dzip(payload, source_name)
        if extracted is not None:
            return extracted
        try:
            return zlib.decompress(payload[12:])
        except zlib.error as exc:
            raise ValueError(f"Failed to decompress DZip payload {source_name}: {exc}") from exc
    try:
        return gzip.decompress(payload)
    except OSError as exc:
        raise ValueError(f"Failed to decompress demo payload {source_name}: {exc}") from exc


def extract_demo_bytes(
    row: Mapping[str, object],
    remote_username: str = "guest",
    remote_password: str = "guest",
    *,
    remote_prefix: str = "",
) -> bytes:
    extracted_path = str(row.get("extracted_dem_path", ""))
    if extracted_path:
        payload = _read_storage_bytes(
            row,
            extracted_path,
            remote_username,
            remote_password,
            remote_prefix=remote_prefix,
        )
        return decompress_demo_payload(payload, source_name=extracted_path)

    local_path = str(row.get("local_path", ""))
    if not local_path:
        raise ValueError("Manifest row is missing local_path")

    payload = _read_storage_bytes(
        row,
        local_path,
        remote_username,
        remote_password,
        remote_prefix=remote_prefix,
    )
    suffix = Path(local_path).suffix.lower()
    if suffix in {".dz", ".dem"}:
        return decompress_demo_payload(payload, source_name=local_path)
    raise ValueError(f"Unsupported manifest payload {local_path}")


def _extract_demo_bytes(row: Mapping[str, object], remote_username: str, remote_password: str) -> bytes:
    return extract_demo_bytes(row, remote_username=remote_username, remote_password=remote_password)


def _safe_stem(row: Mapping[str, object]) -> str:
    source = str(row.get("extracted_dem_path", "")) or str(row.get("local_path", "")) or str(row.get("url", "demo"))
    leaf = source.replace("\\", "/").split("/")[-1]
    stem = Path(leaf).stem or "demo"
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def materialize_corpus_subset(
    manifest_path: str | Path,
    output_dir: str | Path,
    map_id: str,
    remote_username: str = "guest",
    remote_password: str = "guest",
    limit: int | None = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    rows = filter_manifest_rows(load_manifest_rows(manifest_path), map_id=map_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    seen_hashes: set[str] = set()

    for row in rows:
        sha256 = str(row.get("sha256", ""))
        if sha256 and sha256 in seen_hashes:
            continue
        if limit is not None and len(manifest_rows) >= limit:
            break

        try:
            demo_bytes = _extract_demo_bytes(row, remote_username=remote_username, remote_password=remote_password)
        except Exception as exc:
            failure_rows.append(
                {
                    "url": str(row.get("url", "")),
                    "sha256": sha256,
                    "error": str(exc),
                }
            )
            continue

        digest = sha256 or f"materialized_{len(manifest_rows):06d}"
        target = output / f"{len(manifest_rows):04d}_{_safe_stem(row)}_{digest[:12]}.dem"
        if overwrite or not target.exists():
            target.write_bytes(demo_bytes)

        manifest_rows.append(
            {
                "map_id": map_id.lower(),
                "source_url": str(row.get("url", "")),
                "sha256": sha256,
                "materialized_path": str(target),
                "storage_backend": str(row.get("storage_backend", "local")),
                "storage_root": str(row.get("storage_root", "")),
                "local_path": str(row.get("local_path", "")),
                "extracted_dem_path": str(row.get("extracted_dem_path", "")),
            }
        )
        if sha256:
            seen_hashes.add(sha256)

    manifest_out = output / "materialized_manifest.ndjson"
    failures_out = output / "materialize_failures.ndjson"
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with failures_out.open("w", encoding="utf-8") as handle:
        for row in failure_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "map_id": map_id.lower(),
        "manifest_path": str(Path(manifest_path)),
        "output_dir": str(output),
        "requested_rows": len(rows),
        "materialized_demos": len(manifest_rows),
        "failed_rows": len(failure_rows),
        "materialized_manifest": str(manifest_out),
        "failures_manifest": str(failures_out),
    }
    (output / "materialize_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
