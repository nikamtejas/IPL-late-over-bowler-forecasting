"""Stage: download. Fetch the Cricsheet IPL zip into data/raw (skipped if already present).

Raw data is never modified: to refresh it, delete the zip and its manifest by hand.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config, schemas

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    cfg = config.load_config()["download"]
    dest = config.path("raw_zip")
    manifest = config.path("download_manifest")

    if dest.exists():
        log.info("%s already exists; skipping download", dest)
        if not manifest.exists():
            # File was placed by hand: record what we have, but say we didn't download it.
            _write_manifest(manifest, cfg["url"], dest, note="pre-existing file; download date unknown")
        _check_manifest(manifest, dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("downloading %s", cfg["url"])
    with requests.get(cfg["url"], stream=True, timeout=cfg["timeout_seconds"]) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(tmp) as zf:
        bad = zf.testzip()
        assert bad is None, f"corrupt member in downloaded zip: {bad}"
    tmp.rename(dest)
    _write_manifest(manifest, cfg["url"], dest, note="downloaded")
    log.info("saved %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def _write_manifest(manifest: Path, url: str, dest: Path, note: str) -> None:
    schemas.write_json(
        {
            "url": url,
            "file": dest.name,
            "sha256": sha256_of(dest),
            "size_bytes": dest.stat().st_size,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
        },
        schemas.DOWNLOAD_MANIFEST_KEYS,
        manifest,
    )


def _check_manifest(manifest: Path, dest: Path) -> None:
    m = schemas.read_json(manifest, schemas.DOWNLOAD_MANIFEST_KEYS)
    actual = sha256_of(dest)
    if m["sha256"] != actual:
        raise RuntimeError(
            f"{dest} hash {actual[:12]} does not match manifest {m['sha256'][:12]}; "
            "raw data must not change. Delete both files and re-run `make download`."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    download()
