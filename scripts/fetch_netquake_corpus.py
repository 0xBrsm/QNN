#!/usr/bin/env python3
"""Fetch a large NetQuake demo corpus from SDA and Quake Terminus.

Primary storage can be an SMB share; if free space drops below a threshold,
new downloads are redirected to a local fallback directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

try:
    import smbclient
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    smbclient = None

from quake_ai.data.corpus import decompress_demo_payload

USER_AGENT = "quake-ai-corpus-fetcher/0.5 (+netquake-only, multi-source, storage-failover)"

NETQUAKE_EXTS = (".dem", ".dz")
QW_EXTS = (".qwd", ".mvd", ".qwz")
ARCHIVE_EXTS = (".zip",)
DOWNLOADABLE_EXTS = NETQUAKE_EXTS + ARCHIVE_EXTS

SDA_HOST = "speeddemosarchive.com"
SDA_ROOT_MKT = "https://speeddemosarchive.com/quake/mkt.pl"
SDA_CONTEST_INDEX = "https://speeddemosarchive.com/quake/contests/"
SDA_QDQ_MOVIES_INDEX = "https://speeddemosarchive.com/quake/qdq/movies/"
SDA_QUAKE_LIGHT_ZIP = "https://speeddemosarchive.com/quake/downloads/quake-light.zip"
SDA_ARCHIVE_MAPS_ZIP = "https://archive.org/download/SDA_Quake_demos/maps.zip"
SDA_ARCHIVE_COLLECTION_MAPS_ZIP = "https://archive.org/download/Quake_SDA_collection/maps.zip"

ARCHIVE_ORG_HOSTS = {"archive.org", "www.archive.org"}
ARCHIVE_ORG_DEMO_ARCHIVES = [
    "https://archive.org/download/idgames-2-archive/demos.zip",
    SDA_ARCHIVE_MAPS_ZIP,
    SDA_ARCHIVE_COLLECTION_MAPS_ZIP,
]
ARCHIVE_ORG_DEMO_ARCHIVE_PATHS = {urlparse(url).path.lower() for url in ARCHIVE_ORG_DEMO_ARCHIVES}

IGMDB_HOST = "demos.igmdb.org"
IGMDB_CHTV_NETQUAKE_INDEX = "http://demos.igmdb.org/ChallengeTV/demostorage/Quake%201/netquake/"
IGMDB_CHTV_QUAKE1_DIRS = [
    "http://demos.igmdb.org/ChallengeTV/demostorage/Quake%201/netquake/",
    "http://demos.igmdb.org/ChallengeTV/demostorage/Quake%201/frag1/",
    "http://demos.igmdb.org/ChallengeTV/demostorage/Quake%201/reptile/",
    "http://demos.igmdb.org/ChallengeTV/demostorage/Quake%201/thresh/",
    "http://demos.igmdb.org/ChallengeTV/demostorage/Quake%201/entropy/",
]
IGMDB_CHTV_LEGACY_NQ_DIRS = [
    "http://demos.igmdb.org/ChallengeTV/chtv.quakeworld.nu/netquake/",
]

QT_HOSTS = {"www.quaketerminus.com", "quaketerminus.com"}
QT_METHOS_INDEX = "https://www.quaketerminus.com/hosted/methosq_demos/qdemos.htm"
QUADDICTED_DEMOS_INDEX = "https://www.quaddicted.com/files/demos/"
QUAKETASTIC_HOSTS = {"www.quaketastic.com", "quaketastic.com"}
QUAKETASTIC_DEMOS_INDEX = "https://www.quaketastic.com/files/demos/"

IDGAMES2_MIRRORS = [
    "https://www.quaddicted.com/files/idgames2/demos/",
    "https://www.gamers.org/pub/idgames2/demos/",
    "https://mirror.braindrainlan.nu/pub/idgames2/demos/",
    "https://ftpmirror1.infania.net/pub/idgames2/demos/",
    "https://ftp.fu-berlin.de/pc/games/idgames2/demos/",
]

HREF_RE = re.compile(r"href\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
LEVEL_ID_RE = re.compile(r"mkt\.pl\?level:([a-z0-9_\-]+)", re.IGNORECASE)
QT_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
QT_DEMO_ZIP_RE = re.compile(r"href\s*=\s*['\"](demos/[^'\"]+\.zip)['\"]", re.IGNORECASE)
QT_MODE_RE = re.compile(r"<td[^>]*>\s*<p[^>]*align\s*=\s*['\"]center['\"][^>]*>\s*(NQ\d*|QW\d*)\s*</td>", re.IGNORECASE)
QT_SECTION_RE = re.compile(r"<a\s+name=['\"]([A-Za-z0-9._-]+)['\"]", re.IGNORECASE)
SDA_QDQ_PAGE_RE = re.compile(r"/quake/qdq/movies/[^/?#]+\.html$", re.IGNORECASE)

CANONICAL_NETQUAKE_MAP_IDS: set[str] = {
    *(f"e1m{i}" for i in range(1, 9)),
    *(f"e2m{i}" for i in range(1, 8)),
    *(f"e3m{i}" for i in range(1, 8)),
    *(f"e4m{i}" for i in range(1, 9)),
    "end",
    *(f"dm{i}" for i in range(1, 7)),
}


@dataclass(slots=True)
class StoredRef:
    backend: str
    relative_path: str


@dataclass(slots=True)
class FetchResult:
    url: str
    status: str
    content_type: str
    size: int
    sha256: str
    storage_backend: str
    storage_root: str
    local_path: str
    extracted_dem_path: str


class StorageRouter:
    def __init__(
        self,
        local_root: Path,
        remote_share: Optional[str],
        remote_username: str,
        remote_password: str,
        remote_free_threshold: float,
        space_check_every: int,
    ) -> None:
        self.local_root = local_root
        self.local_root.mkdir(parents=True, exist_ok=True)

        self.remote_share = remote_share
        self.remote_username = remote_username
        self.remote_password = remote_password
        self.remote_free_threshold = remote_free_threshold
        self.space_check_every = max(1, space_check_every)

        self.remote_available = False
        self.remote_latched_to_local = False
        self.last_remote_free_ratio: Optional[float] = None
        self.writes_since_remote_check = 0

        if self.remote_share:
            self._init_remote_session()

    def _init_remote_session(self) -> None:
        assert self.remote_share is not None
        if smbclient is None:
            self.remote_available = False
            self.remote_latched_to_local = True
            return
        host = self._remote_host(self.remote_share)
        try:
            smbclient.ClientConfig(require_secure_negotiate=False)
            smbclient.register_session(
                host,
                username=self.remote_username,
                password=self.remote_password,
                encrypt=False,
                require_signing=False,
            )
            self._ensure_remote_dir("")
            self._ensure_remote_dir("raw")
            self._ensure_remote_dir("dem")
            self._ensure_remote_dir("meta")
            self.remote_available = True
        except Exception:
            self.remote_available = False
            self.remote_latched_to_local = True

    @staticmethod
    def _remote_host(share: str) -> str:
        stripped = share.lstrip("\\")
        return stripped.split("\\", 1)[0]

    def _remote_join(self, relative_path: str) -> str:
        assert self.remote_share is not None
        rel = relative_path.replace("/", "\\").strip("\\")
        if not rel:
            return self.remote_share
        return self.remote_share.rstrip("\\") + "\\" + rel

    def _ensure_remote_dir(self, relative_dir: str) -> None:
        if smbclient is None:
            return
        path = self._remote_join(relative_dir)
        if not smbclient.path.exists(path):
            smbclient.makedirs(path, exist_ok=True)

    def _remote_free_ratio(self) -> Optional[float]:
        if smbclient is None or not self.remote_share or not self.remote_available:
            return None
        try:
            statv = smbclient.stat_volume(self.remote_share)
            total = float(statv.total_size)
            free = float(statv.actual_available_size)
            if total <= 0:
                return None
            return free / total
        except Exception:
            return None

    def _maybe_flip_to_local(self, force: bool = False) -> None:
        if not self.remote_share or not self.remote_available or self.remote_latched_to_local:
            return

        if not force and self.writes_since_remote_check < self.space_check_every:
            return

        self.writes_since_remote_check = 0
        ratio = self._remote_free_ratio()
        self.last_remote_free_ratio = ratio
        if ratio is None or ratio < self.remote_free_threshold:
            self.remote_latched_to_local = True

    def choose_backend(self) -> str:
        self._maybe_flip_to_local(force=False)
        if self.remote_share and self.remote_available and not self.remote_latched_to_local:
            return "remote"
        return "local"

    def write_bytes(self, relative_path: str, content: bytes) -> StoredRef:
        backend = self.choose_backend()

        if backend == "remote":
            if smbclient is None:
                backend = "local"
            else:
                remote_path = self._remote_join(relative_path)
                parent = remote_path.rsplit("\\", 1)[0]
                if not smbclient.path.exists(parent):
                    smbclient.makedirs(parent, exist_ok=True)
                with smbclient.open_file(remote_path, mode="wb") as handle:
                    handle.write(content)
                self.writes_since_remote_check += 1
                self._maybe_flip_to_local(force=False)
                return StoredRef(backend="remote", relative_path=relative_path)

        target = self.local_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return StoredRef(backend="local", relative_path=relative_path)

    def root_for_backend(self, backend: str) -> str:
        if backend == "remote" and self.remote_share:
            return self.remote_share
        return str(self.local_root)


class SDACollector:
    def __init__(
        self,
        out_root: Path,
        router: StorageRouter,
        focus_profile: str,
        max_pages: int,
        max_downloads: int,
        max_total_bytes: int,
        sleep_seconds: float,
    ) -> None:
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.router = router
        self.focus_profile = focus_profile
        self.max_pages = max_pages
        self.max_downloads = max_downloads
        self.max_total_bytes = max_total_bytes
        self.sleep_seconds = sleep_seconds

        self.downloaded_urls: set[str] = set()
        self.hash_to_raw: dict[str, StoredRef] = {}
        self.hash_to_dem: dict[str, StoredRef] = {}

        self.pages_scanned = 0
        self.downloads = 0
        self.total_bytes = 0

        self.meta_root = out_root / "meta"
        self.meta_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.meta_root / "download_manifest.ndjson"
        self.errors_path = self.meta_root / "errors.ndjson"

        if self.focus_profile == "canonical-netquake":
            self.allowed_map_ids = set(CANONICAL_NETQUAKE_MAP_IDS)
        else:
            self.allowed_map_ids = None

        self._load_existing_manifest()

    def _load_existing_manifest(self) -> None:
        if not self.manifest_path.exists():
            return

        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            url = str(row.get("url", ""))
            sha = str(row.get("sha256", ""))
            rel_raw = str(row.get("local_path", ""))
            rel_dem = str(row.get("extracted_dem_path", ""))
            backend = str(row.get("storage_backend", "local"))
            size = int(row.get("size", 0) or 0)
            status = str(row.get("status", ""))

            if url:
                self.downloaded_urls.add(url)
            if sha and rel_raw:
                self.hash_to_raw[sha] = StoredRef(backend=backend, relative_path=rel_raw)
            if sha and rel_dem:
                self.hash_to_dem[sha] = StoredRef(backend=backend, relative_path=rel_dem)

            if status == "downloaded":
                self.downloads += 1
                self.total_bytes += max(size, 0)

    @staticmethod
    def _normalize_url(base_url: str, href: str) -> str:
        href = unescape(href.strip())
        if not href or href.startswith("#"):
            return ""
        if href.lower().startswith("javascript:"):
            return ""
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in {"http", "https"}:
            return ""
        return parsed._replace(fragment="").geturl()

    @staticmethod
    def _is_netquake_demo_url(url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if any(path.endswith(ext) for ext in QW_EXTS):
            return False
        if not any(path.endswith(ext) for ext in DOWNLOADABLE_EXTS):
            return False

        if host == SDA_HOST:
            if "/quake/demos/" in path or "/quake/contests/demos/" in path:
                return True
            if path.startswith("/quake/qdq/demos/"):
                return True
            if path.startswith("/quake/qdq/hobby/") or path.startswith("/quake/projects/hobby/"):
                return True
            if path == "/quake/downloads/quake-light.zip":
                return True
            return False

        if host == IGMDB_HOST:
            return path.startswith("/challengetv/demostorage/quake%201/") or path.startswith("/challengetv/chtv.quakeworld.nu/netquake/")

        if host in ARCHIVE_ORG_HOSTS and path in ARCHIVE_ORG_DEMO_ARCHIVE_PATHS:
            return True

        for root in IDGAMES2_MIRRORS:
            root_parsed = urlparse(root)
            if host == root_parsed.netloc.lower() and path.startswith(root_parsed.path.lower()):
                return True

        qd = urlparse(QUADDICTED_DEMOS_INDEX)
        if host == qd.netloc.lower() and path.startswith(qd.path.lower()):
            return True

        if host in QUAKETASTIC_HOSTS and path.startswith("/files/demos/"):
            return True

        return False

    @staticmethod
    def _map_id_from_name(name: str) -> Optional[str]:
        if not name:
            return None
        filename = name.split("/")[-1]
        if "." not in filename:
            return None
        stem = filename.rsplit(".", 1)[0]
        lowered = stem.lower()
        prefix = lowered.split("_", 1)[0]
        match = re.match(r"^(e[1-4]m[1-8]|dm[1-6]|end)", prefix)
        if match:
            return match.group(1)
        match = re.search(r"(?:^|[^a-z0-9])(e[1-4]m[1-8]|dm[1-6]|end)(?:[^a-z0-9]|$)", lowered)
        if match:
            return match.group(1)
        return prefix

    @classmethod
    def _map_id_from_demo_url(cls, url: str) -> Optional[str]:
        path = urlparse(url).path
        if not path:
            return None
        return cls._map_id_from_name(path)

    @staticmethod
    def _is_aggregate_archive_url(url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.lower()
        host = parsed.netloc.lower()
        if host == SDA_HOST:
            return (
                path.startswith("/quake/qdq/demos/")
                or path.startswith("/quake/qdq/hobby/")
                or path.startswith("/quake/projects/hobby/")
                or path == "/quake/downloads/quake-light.zip"
            )
        if host == IGMDB_HOST:
            return path.startswith("/challengetv/demostorage/quake%201/") or path.startswith("/challengetv/chtv.quakeworld.nu/netquake/")
        if host in QUAKETASTIC_HOSTS:
            return path.startswith("/files/demos/")
        if host in ARCHIVE_ORG_HOSTS and path in ARCHIVE_ORG_DEMO_ARCHIVE_PATHS:
            return True

        for root in IDGAMES2_MIRRORS:
            root_parsed = urlparse(root)
            if host == root_parsed.netloc.lower() and path.startswith(root_parsed.path.lower()):
                return True

        qd = urlparse(QUADDICTED_DEMOS_INDEX)
        if host == qd.netloc.lower() and path.startswith(qd.path.lower()):
            return True

        return False

    def _is_allowed_demo_url(self, url: str) -> bool:
        if self.allowed_map_ids is None:
            return True
        map_id = self._map_id_from_demo_url(url)
        if map_id is not None and map_id in self.allowed_map_ids:
            return True
        if url.lower().endswith(ARCHIVE_EXTS) and self._is_aggregate_archive_url(url):
            return True
        return False

    @staticmethod
    def _extract_hrefs(html_text: str) -> Iterable[str]:
        for match in HREF_RE.finditer(html_text):
            yield match.group(1)

    @staticmethod
    def _extract_level_ids(html_text: str) -> list[str]:
        # Preserve first-seen order so we can retain the source's intended
        # grouping (id levels first) rather than forcing pure lexicographic order.
        seen: set[str] = set()
        ordered: list[str] = []
        for match in LEVEL_ID_RE.finditer(html_text):
            level_id = match.group(1).lower()
            if level_id in seen:
                continue
            seen.add(level_id)
            ordered.append(level_id)
        return ordered

    @staticmethod
    def _level_priority(level_id: str) -> tuple[int, str]:
        # Prioritize canonical id levels early so initial training corpora are
        # useful before the full long-tail custom map crawl completes.
        if re.fullmatch(r"e[1-4]m[1-8]", level_id):
            return (0, level_id)
        if level_id in {"end", "ep1", "ep2", "ep3", "ep4", "id1"}:
            return (1, level_id)
        if re.fullmatch(r"hip[1-3]m[1-6]|hipdm1|hipend", level_id):
            return (2, level_id)
        if re.fullmatch(r"r[12]m[1-8]|doe[12]|doe", level_id):
            return (3, level_id)
        return (4, level_id)

    @staticmethod
    def _relative_path_for_url(prefix: str, url: str, suffix: Optional[str] = None) -> str:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            parts = ["index"]
        safe_parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in parts]
        rel = Path(prefix) / parsed.netloc / Path(*safe_parts)
        if suffix is not None:
            rel = rel.with_suffix(suffix)
        return rel.as_posix()

    def _write_jsonl(self, path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _record_error(self, url: str, error: str) -> None:
        self._write_jsonl(self.errors_path, {"url": url, "error": error})

    def _request(self, url: str) -> tuple[bytes, str]:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=45) as resp:
            content = resp.read()
            ctype = resp.headers.get("Content-Type", "application/octet-stream")
            return content, ctype

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value)
        return cleaned or "demo.dem"

    def _zip_member_relative_dem_path(self, source_url: str, member_name: str) -> str:
        base = Path(self._relative_path_for_url("dem", source_url)).with_suffix("")
        leaf = self._safe_filename(Path(member_name).name)
        leaf = Path(leaf).with_suffix(".dem").name
        return (base.parent / f"{base.name}__{leaf}").as_posix()

    def _zip_member_relative_dz_path(self, source_url: str, member_name: str) -> str:
        base = Path(self._relative_path_for_url("dem", source_url)).with_suffix("")
        leaf = self._safe_filename(Path(member_name).name)
        leaf = Path(leaf).with_suffix(".dz").name
        return (base.parent / f"{base.name}__{leaf}").as_posix()

    def _is_allowed_member_name(self, member_name: str) -> bool:
        if self.allowed_map_ids is None:
            return True
        map_id = self._map_id_from_name(member_name)
        return map_id is not None and map_id in self.allowed_map_ids

    def _iter_zip_demo_members(
        self,
        blob: bytes,
        member_prefix: str = "",
        depth: int = 0,
        max_depth: int = 1,
    ) -> list[tuple[str, bytes]]:
        members: list[tuple[str, bytes]] = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except (zipfile.BadZipFile, RuntimeError, OSError):
            return members

        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = member.filename.replace("\\", "/")
            member_lower = member_path.lower()
            if any(member_lower.endswith(ext) for ext in QW_EXTS):
                continue

            if member_lower.endswith(".zip") and depth < max_depth:
                try:
                    nested_blob = zf.read(member)
                except Exception:
                    continue
                nested_prefix = f"{member_prefix}{self._safe_filename(Path(member_path).stem)}__"
                members.extend(
                    self._iter_zip_demo_members(
                        nested_blob,
                        member_prefix=nested_prefix,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )
                continue

            if not any(member_lower.endswith(ext) for ext in NETQUAKE_EXTS):
                continue
            if not self._is_allowed_member_name(member_path):
                continue

            try:
                member_bytes = zf.read(member)
            except Exception:
                continue

            leaf = self._safe_filename(Path(member_path).name)
            safe_name = f"{member_prefix}{leaf}" if member_prefix else leaf
            members.append((safe_name, member_bytes))

        return members

    def _save_content(self, url: str, content: bytes, sha: str) -> tuple[StoredRef, Optional[StoredRef]]:
        if sha in self.hash_to_raw:
            raw_ref = self.hash_to_raw[sha]
            dem_ref = self.hash_to_dem.get(sha)
            return raw_ref, dem_ref

        raw_rel = self._relative_path_for_url("raw", url)
        raw_ref = self.router.write_bytes(raw_rel, content)
        self.hash_to_raw[sha] = raw_ref

        dem_ref: Optional[StoredRef] = None
        if url.lower().endswith(".dz"):
            try:
                dem_bytes = decompress_demo_payload(content, source_name=url)
                dem_rel = self._relative_path_for_url("dem", url, suffix=".dem")
                dem_ref = self.router.write_bytes(dem_rel, dem_bytes)
                self.hash_to_dem[sha] = dem_ref
            except ValueError:
                dz_rel = self._relative_path_for_url("dem", url, suffix=".dz")
                dem_ref = self.router.write_bytes(dz_rel, content)
                self.hash_to_dem[sha] = dem_ref
        elif url.lower().endswith(".dem"):
            dem_rel = self._relative_path_for_url("dem", url)
            dem_ref = self.router.write_bytes(dem_rel, content)
            self.hash_to_dem[sha] = dem_ref
        elif url.lower().endswith(ARCHIVE_EXTS):
            try:
                first_dem_ref: Optional[StoredRef] = None
                for member_name, member_bytes in self._iter_zip_demo_members(content):
                    if member_name.lower().endswith(".dem"):
                        dem_rel = self._zip_member_relative_dem_path(url, member_name)
                        written = self.router.write_bytes(dem_rel, member_bytes)
                    else:
                        try:
                            dem_bytes = decompress_demo_payload(member_bytes, source_name=member_name)
                            dem_rel = self._zip_member_relative_dem_path(url, member_name)
                            written = self.router.write_bytes(dem_rel, dem_bytes)
                        except ValueError:
                            dz_rel = self._zip_member_relative_dz_path(url, member_name)
                            written = self.router.write_bytes(dz_rel, member_bytes)
                    if first_dem_ref is None:
                        first_dem_ref = written
                if first_dem_ref is not None:
                    dem_ref = first_dem_ref
                    self.hash_to_dem[sha] = dem_ref
            except (zipfile.BadZipFile, RuntimeError, OSError):
                dem_ref = None

        return raw_ref, dem_ref

    def _record_download(self, result: FetchResult) -> None:
        payload = {
            "url": result.url,
            "status": result.status,
            "content_type": result.content_type,
            "size": result.size,
            "sha256": result.sha256,
            "storage_backend": result.storage_backend,
            "storage_root": result.storage_root,
            "local_path": result.local_path,
            "extracted_dem_path": result.extracted_dem_path,
        }
        self._write_jsonl(self.manifest_path, payload)

    def _extract_catalog_pages(self, base_url: str, html_text: str) -> list[str]:
        pages: set[str] = set()
        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(base_url, href)
            if not full:
                continue
            p = urlparse(full)
            if p.netloc != SDA_HOST:
                continue

            path = p.path.lower()
            query = p.query.lower()

            if path.startswith("/quake/contests/") and path.endswith(".html"):
                pages.add(full)
                continue

            if path.endswith("/quake/mkt.pl"):
                if not query:
                    continue
                if query.startswith("player:"):
                    continue
                if query.startswith("level:"):
                    continue
                pages.add(full)
        return sorted(pages)

    def _extract_qt_nq_archives(self, index_url: str, html_text: str) -> list[str]:
        section_markers: list[tuple[int, str]] = []
        for match in QT_SECTION_RE.finditer(html_text):
            section_markers.append((match.start(), match.group(1).lower().strip(".")))

        archives: list[str] = []
        seen: set[str] = set()
        marker_idx = 0
        for table_match in QT_TABLE_RE.finditer(html_text):
            block = table_match.group(1)

            href_match = QT_DEMO_ZIP_RE.search(block)
            if not href_match:
                continue

            mode_tokens = [token.upper() for token in QT_MODE_RE.findall(block)]
            mode = next((token for token in mode_tokens if token.startswith(("NQ", "QW"))), "")
            if not mode or not mode.startswith("NQ"):
                continue

            while marker_idx + 1 < len(section_markers) and section_markers[marker_idx + 1][0] <= table_match.start():
                marker_idx += 1
            map_id = section_markers[marker_idx][1] if section_markers else ""
            if self.allowed_map_ids is not None and map_id not in self.allowed_map_ids:
                continue

            full = self._normalize_url(index_url, href_match.group(1))
            if not full:
                continue

            parsed = urlparse(full)
            if parsed.netloc.lower() not in QT_HOSTS:
                continue
            if "/hosted/methosq_demos/demos/" not in parsed.path.lower():
                continue

            if full in seen:
                continue
            seen.add(full)
            archives.append(full)
        return archives

    def _pick_live_source(self, candidates: list[str]) -> tuple[Optional[str], str]:
        for url in candidates:
            try:
                html = self._request(url)[0].decode("latin-1", errors="ignore")
                self.pages_scanned += 1
                return url, html
            except Exception as exc:
                self._record_error(url, f"seed_fetch_failed: {exc}")
        return None, ""

    def _extract_idgames_pages(self, root_url: str, html_text: str) -> list[str]:
        root = urlparse(root_url)
        pages: set[str] = set()
        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(root_url, href)
            if not full:
                continue
            p = urlparse(full)
            if p.netloc.lower() != root.netloc.lower():
                continue
            if p.query:
                continue
            if not p.path.endswith("/"):
                continue
            if not p.path.startswith(root.path):
                continue
            if p.path.rstrip("/") == root.path.rstrip("/"):
                continue
            pages.add(full)
        return sorted(pages)

    def _extract_quaddicted_demo_urls(self, index_url: str, html_text: str) -> list[str]:
        index = urlparse(index_url)
        urls: list[str] = []
        seen: set[str] = set()
        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(index_url, href)
            if not full:
                continue
            p = urlparse(full)
            if p.netloc.lower() != index.netloc.lower():
                continue
            if not p.path.startswith(index.path):
                continue
            if p.path.endswith("/"):
                continue
            if not self._is_netquake_demo_url(full):
                continue
            if full in seen:
                continue
            seen.add(full)
            urls.append(full)
        return urls

    def _extract_quaketastic_demo_urls(self, index_url: str, html_text: str) -> tuple[list[str], list[str]]:
        files: list[str] = []
        file_seen: set[str] = set()
        subdirs: list[str] = []
        subdir_seen: set[str] = set()

        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(index_url, href)
            if not full:
                continue
            parsed = urlparse(full)
            if parsed.netloc.lower() not in QUAKETASTIC_HOSTS:
                continue
            if not parsed.path.lower().startswith("/files/demos/"):
                continue

            if parsed.path.endswith("/"):
                if parsed.path.rstrip("/") == "/files/demos":
                    continue
                if full in subdir_seen:
                    continue
                subdir_seen.add(full)
                subdirs.append(full)
                continue

            if not self._is_netquake_demo_url(full):
                continue
            if full in file_seen:
                continue
            file_seen.add(full)
            files.append(full)

        return files, subdirs

    def _extract_sda_qdq_movie_pages(self, index_url: str, html_text: str) -> list[str]:
        pages: set[str] = set()
        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(index_url, href)
            if not full:
                continue
            parsed = urlparse(full)
            if parsed.netloc.lower() != SDA_HOST:
                continue
            if not SDA_QDQ_PAGE_RE.search(parsed.path):
                continue
            pages.add(full)
        return sorted(pages)

    def _extract_sda_qdq_demo_urls(self, page_url: str, html_text: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(page_url, href)
            if not full:
                continue
            if not self._is_netquake_demo_url(full):
                continue
            parsed = urlparse(full)
            if parsed.netloc.lower() != SDA_HOST:
                continue
            path = parsed.path.lower()
            if path.startswith("/quake/qdq/demos/") or path.startswith("/quake/qdq/hobby/") or path.startswith("/quake/projects/hobby/"):
                if full in seen:
                    continue
                seen.add(full)
                urls.append(full)
        return urls

    def _extract_igmdb_chtv_urls(self, index_url: str, html_text: str) -> list[str]:
        index = urlparse(index_url)
        urls: list[str] = []
        seen: set[str] = set()
        for href in self._extract_hrefs(html_text):
            full = self._normalize_url(index_url, href)
            if not full:
                continue
            parsed = urlparse(full)
            if parsed.netloc.lower() != index.netloc.lower():
                continue
            if not parsed.path.lower().startswith(index.path.lower()):
                continue
            if parsed.path.endswith("/"):
                continue
            if not self._is_netquake_demo_url(full):
                continue
            if full in seen:
                continue
            seen.add(full)
            urls.append(full)
        return urls

    def _download_demo(self, url: str) -> None:
        if url in self.downloaded_urls:
            return
        if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
            return

        try:
            blob, demo_ct = self._request(url)
        except Exception as exc:
            self._record_error(url, f"download_failed: {exc}")
            return

        sha = self._sha256(blob)
        raw_ref, dem_ref = self._save_content(url, blob, sha)

        if url.lower().endswith(ARCHIVE_EXTS) and dem_ref is None:
            self.downloaded_urls.add(url)
            self._record_download(
                FetchResult(
                    url=url,
                    status="skipped_no_netquake_dem",
                    content_type=demo_ct,
                    size=len(blob),
                    sha256=sha,
                    storage_backend=raw_ref.backend,
                    storage_root=self.router.root_for_backend(raw_ref.backend),
                    local_path=raw_ref.relative_path,
                    extracted_dem_path="",
                )
            )
            return

        self.downloaded_urls.add(url)
        self.downloads += 1
        self.total_bytes += len(blob)

        self._record_download(
            FetchResult(
                url=url,
                status="downloaded",
                content_type=demo_ct,
                size=len(blob),
                sha256=sha,
                storage_backend=raw_ref.backend,
                storage_root=self.router.root_for_backend(raw_ref.backend),
                local_path=raw_ref.relative_path,
                extracted_dem_path=dem_ref.relative_path if dem_ref else "",
            )
        )

    def _process_page_for_demos(self, page_url: str, enforce_page_cap: bool = True) -> None:
        if enforce_page_cap and self.pages_scanned >= self.max_pages:
            return

        try:
            content, _ = self._request(page_url)
        except Exception as exc:
            self._record_error(page_url, f"page_fetch_failed: {exc}")
            return

        self.pages_scanned += 1

        text = content.decode("latin-1", errors="ignore")
        for href in self._extract_hrefs(text):
            full = self._normalize_url(page_url, href)
            if not full:
                continue
            if not self._is_netquake_demo_url(full):
                continue
            if not self._is_allowed_demo_url(full):
                continue
            self._download_demo(full)
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break

    def run(self) -> dict:
        try:
            mkt_html = self._request(SDA_ROOT_MKT)[0].decode("latin-1", errors="ignore")
        except Exception as exc:
            self._record_error(SDA_ROOT_MKT, f"seed_fetch_failed: {exc}")
            mkt_html = ""

        try:
            contests_html = self._request(SDA_CONTEST_INDEX)[0].decode("latin-1", errors="ignore")
        except Exception as exc:
            self._record_error(SDA_CONTEST_INDEX, f"seed_fetch_failed: {exc}")
            contests_html = ""

        _, qt_html = self._pick_live_source([QT_METHOS_INDEX])
        _, quaddicted_html = self._pick_live_source([QUADDICTED_DEMOS_INDEX])
        _, quaketastic_html = self._pick_live_source([QUAKETASTIC_DEMOS_INDEX])
        idgames_root, idgames_root_html = self._pick_live_source(IDGAMES2_MIRRORS)
        _, qdq_movies_html = self._pick_live_source([SDA_QDQ_MOVIES_INDEX])
        igmdb_chtv_urls: list[str] = []
        igmdb_seen: set[str] = set()
        for igmdb_dir in IGMDB_CHTV_QUAKE1_DIRS:
            _, igmdb_html = self._pick_live_source([igmdb_dir])
            for candidate in self._extract_igmdb_chtv_urls(igmdb_dir, igmdb_html):
                if candidate in igmdb_seen:
                    continue
                igmdb_seen.add(candidate)
                igmdb_chtv_urls.append(candidate)
        igmdb_legacy_nq_urls: list[str] = []
        for igmdb_dir in IGMDB_CHTV_LEGACY_NQ_DIRS:
            _, igmdb_html = self._pick_live_source([igmdb_dir])
            for candidate in self._extract_igmdb_chtv_urls(igmdb_dir, igmdb_html):
                if candidate in igmdb_seen:
                    continue
                igmdb_seen.add(candidate)
                igmdb_legacy_nq_urls.append(candidate)
                igmdb_chtv_urls.append(candidate)

        level_ids = self._extract_level_ids(mkt_html)
        if self.allowed_map_ids is not None:
            level_ids = [level_id for level_id in level_ids if level_id in self.allowed_map_ids]
        prioritized_level_ids = sorted(level_ids, key=self._level_priority)
        level_pages = [f"{SDA_ROOT_MKT}?level:{lid}" for lid in prioritized_level_ids]

        catalog_pages = self._extract_catalog_pages(SDA_ROOT_MKT, mkt_html)
        contest_pages = self._extract_catalog_pages(SDA_CONTEST_INDEX, contests_html)
        qt_nq_archives = self._extract_qt_nq_archives(QT_METHOS_INDEX, qt_html)
        quaddicted_demo_urls = self._extract_quaddicted_demo_urls(QUADDICTED_DEMOS_INDEX, quaddicted_html)
        quaketastic_demo_urls, quaketastic_subdirs = self._extract_quaketastic_demo_urls(QUAKETASTIC_DEMOS_INDEX, quaketastic_html)
        idgames_pages = self._extract_idgames_pages(idgames_root, idgames_root_html) if idgames_root else []
        qdq_movie_pages = self._extract_sda_qdq_movie_pages(SDA_QDQ_MOVIES_INDEX, qdq_movies_html)

        qdq_demo_urls: list[str] = []
        qdq_seen: set[str] = set()

        def _add_qdq_url(candidate: str) -> None:
            if candidate in qdq_seen:
                return
            qdq_seen.add(candidate)
            qdq_demo_urls.append(candidate)

        for demo_url in self._extract_sda_qdq_demo_urls(SDA_QDQ_MOVIES_INDEX, qdq_movies_html):
            _add_qdq_url(demo_url)

        for page_url in qdq_movie_pages:
            try:
                page_content, _ = self._request(page_url)
            except Exception as exc:
                self._record_error(page_url, f"page_fetch_failed: {exc}")
                continue
            self.pages_scanned += 1
            page_text = page_content.decode("latin-1", errors="ignore")
            for demo_url in self._extract_sda_qdq_demo_urls(page_url, page_text):
                _add_qdq_url(demo_url)

        _add_qdq_url(SDA_QUAKE_LIGHT_ZIP)

        quaketastic_seen: set[str] = set(quaketastic_demo_urls)
        for subdir_url in quaketastic_subdirs:
            try:
                subdir_content, _ = self._request(subdir_url)
            except Exception as exc:
                self._record_error(subdir_url, f"page_fetch_failed: {exc}")
                continue
            self.pages_scanned += 1
            subdir_text = subdir_content.decode("latin-1", errors="ignore")
            subdir_files, _ = self._extract_quaketastic_demo_urls(subdir_url, subdir_text)
            for candidate in subdir_files:
                if candidate in quaketastic_seen:
                    continue
                quaketastic_seen.add(candidate)
                quaketastic_demo_urls.append(candidate)

        page_queue: list[str] = []
        seen_pages: set[str] = set()
        for url in level_pages + catalog_pages + contest_pages:
            if url not in seen_pages:
                seen_pages.add(url)
                page_queue.append(url)

        for seed in [SDA_ROOT_MKT, SDA_CONTEST_INDEX]:
            if seed not in seen_pages:
                seen_pages.add(seed)
                page_queue.append(seed)

        for idx, page_url in enumerate(page_queue):
            if idx >= self.max_pages:
                break
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            self._process_page_for_demos(page_url)
            time.sleep(self.sleep_seconds)

        for demo_url in igmdb_chtv_urls:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            if not self._is_allowed_demo_url(demo_url):
                continue
            self._download_demo(demo_url)
            time.sleep(self.sleep_seconds)

        for demo_url in quaketastic_demo_urls:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            if not self._is_allowed_demo_url(demo_url):
                continue
            self._download_demo(demo_url)
            time.sleep(self.sleep_seconds)

        for demo_url in ARCHIVE_ORG_DEMO_ARCHIVES:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            if not self._is_allowed_demo_url(demo_url):
                continue
            self._download_demo(demo_url)
            time.sleep(self.sleep_seconds)

        for demo_url in quaddicted_demo_urls:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            if not self._is_allowed_demo_url(demo_url):
                continue
            self._download_demo(demo_url)
            time.sleep(self.sleep_seconds)

        for demo_url in qdq_demo_urls:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            if not self._is_allowed_demo_url(demo_url):
                continue
            self._download_demo(demo_url)
            time.sleep(self.sleep_seconds)

        for archive_url in qt_nq_archives:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            self._download_demo(archive_url)
            time.sleep(self.sleep_seconds)

        for page_url in idgames_pages:
            if self.downloads >= self.max_downloads or self.total_bytes >= self.max_total_bytes:
                break
            self._process_page_for_demos(page_url, enforce_page_cap=False)
            time.sleep(self.sleep_seconds)

        summary = {
            "pages_scanned": self.pages_scanned,
            "downloads": self.downloads,
            "total_bytes": self.total_bytes,
            "unique_hashes": len(self.hash_to_raw),
            "focus_profile": self.focus_profile,
            "allowed_map_ids": sorted(self.allowed_map_ids) if self.allowed_map_ids is not None else "all",
            "levels_discovered": len(level_ids),
            "catalog_pages_discovered": len(catalog_pages),
            "contest_pages_discovered": len(contest_pages),
            "idgames_mirror": idgames_root,
            "idgames_pages_discovered": len(idgames_pages),
            "quaddicted_demo_urls_discovered": len(quaddicted_demo_urls),
            "quaketastic_demo_urls_discovered": len(quaketastic_demo_urls),
            "quaketastic_subdirs_discovered": len(quaketastic_subdirs),
            "sda_qdq_movie_pages_discovered": len(qdq_movie_pages),
            "sda_qdq_demo_urls_discovered": len(qdq_demo_urls),
            "igmdb_chtv_netquake_urls_discovered": len(igmdb_chtv_urls),
            "igmdb_chtv_legacy_nq_urls_discovered": len(igmdb_legacy_nq_urls),
            "archive_org_demo_archives_discovered": len(ARCHIVE_ORG_DEMO_ARCHIVES),
            "quaketerminus_nq_archives_discovered": len(qt_nq_archives),
            "remote_share": self.router.remote_share,
            "remote_available": self.router.remote_available,
            "remote_latched_to_local": self.router.remote_latched_to_local,
            "remote_last_free_ratio": self.router.last_remote_free_ratio,
            "manifest": str(self.manifest_path),
            "errors": str(self.errors_path),
        }
        (self.meta_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NetQuake demo corpus from SDA + QdQ + Quake Terminus + idgames mirrors/archive + ChallengeTV + Quaketastic")
    parser.add_argument("--out", default="../artifacts/corpus/netquake", help="Local output/meta root")
    parser.add_argument(
        "--focus-profile",
        choices=["all", "canonical-netquake"],
        default="canonical-netquake",
        help="Restrict crawl to a map profile",
    )
    parser.add_argument("--remote-share", default=r"\\pi.local\nqcorpus\netquake", help="Primary SMB destination root")
    parser.add_argument("--remote-username", default="guest")
    parser.add_argument("--remote-password", default="guest")
    parser.add_argument("--remote-free-threshold", type=float, default=0.10, help="Fallback to local when remote free ratio drops below this")
    parser.add_argument("--space-check-every", type=int, default=25, help="Check remote free space after this many writes")
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--max-downloads", type=int, default=500000)
    parser.add_argument("--max-total-gb", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.005, help="Polite delay between requests")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    router = StorageRouter(
        local_root=out,
        remote_share=args.remote_share,
        remote_username=args.remote_username,
        remote_password=args.remote_password,
        remote_free_threshold=args.remote_free_threshold,
        space_check_every=args.space_check_every,
    )

    collector = SDACollector(
        out_root=out,
        router=router,
        focus_profile=args.focus_profile,
        max_pages=args.max_pages,
        max_downloads=args.max_downloads,
        max_total_bytes=int(args.max_total_gb * 1024 * 1024 * 1024),
        sleep_seconds=max(args.sleep, 0.0),
    )
    summary = collector.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
