import json
import shutil

import pytest
import yaml

from ipl_late_overs import config, download

from .conftest import FIXTURES


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """A copy of config.yaml whose raw paths point into tmp_path."""
    cfg = yaml.safe_load(config.config_path().read_text())
    cfg["paths"]["raw_zip"] = str(tmp_path / "ipl_json.zip")
    cfg["paths"]["download_manifest"] = str(tmp_path / "download_manifest.json")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv(config.CONFIG_ENV_VAR, str(p))
    return cfg


def test_project_config_loads():
    cfg = config.load_config()
    s = cfg["splits"]
    assert max(s["train_seasons"]) < min(s["validation_seasons"]) < min(s["test_seasons"])


def test_overlapping_splits_rejected(tmp_config, tmp_path):
    tmp_config["splits"]["test_seasons"] = [tmp_config["splits"]["validation_seasons"][-1]]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(tmp_config))
    with pytest.raises(AssertionError):
        config._load(p)


def test_download_skips_existing_file_and_records_hash(tmp_config, tmp_path, monkeypatch):
    zip_path = tmp_path / "ipl_json.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", FIXTURES)

    def no_network(*a, **k):
        raise AssertionError("must not download when the file exists")

    monkeypatch.setattr(download.requests, "get", no_network)
    download.download()
    m = json.loads((tmp_path / "download_manifest.json").read_text())
    assert m["sha256"] == download.sha256_of(zip_path)
    assert m["note"].startswith("pre-existing")
    download.download()  # second run: manifest exists and matches


def test_changed_raw_file_fails_loudly(tmp_config, tmp_path, monkeypatch):
    zip_path = tmp_path / "ipl_json.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", FIXTURES)
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: None)
    download.download()
    with open(zip_path, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(RuntimeError, match="does not match manifest"):
        download.download()
