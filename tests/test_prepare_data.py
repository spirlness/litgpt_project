from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

import prepare_data


def test_load_yaml_returns_empty_dict_for_empty_file(tmp_path: Path) -> None:
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("", encoding="utf-8")

    result = prepare_data.load_yaml(yaml_path)

    assert result == {}


def test_load_yaml_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(RuntimeError):
        prepare_data.load_yaml(missing_path)


def test_load_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text("a: [", encoding="utf-8")

    with pytest.raises(RuntimeError):
        prepare_data.load_yaml(yaml_path)


def test_download_file_writes_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "tokenizer.json"
    response = Mock()
    response.iter_content.return_value = [b"abc", b"def"]
    response.raise_for_status.return_value = None

    def fake_get(*_args, **_kwargs):
        return response

    monkeypatch.setattr(prepare_data.requests, "get", fake_get)

    prepare_data.download_file("https://example.com/tokenizer.json", target)

    assert target.read_bytes() == b"abcdef"


def test_download_file_handles_request_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "tokenizer.json"

    def fake_get(*_args, **_kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(prepare_data.requests, "get", fake_get)

    with pytest.raises(RuntimeError):
        prepare_data.download_file("https://example.com/tokenizer.json", target)

    assert not target.exists()


def test_download_file_rejects_empty_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "tokenizer.json"
    response = Mock()
    response.iter_content.return_value = []
    response.raise_for_status.return_value = None

    def fake_get(*_args, **_kwargs):
        return response

    monkeypatch.setattr(prepare_data.requests, "get", fake_get)

    with pytest.raises(RuntimeError):
        prepare_data.download_file("https://example.com/tokenizer.json", target)

    assert not target.exists()


def test_split_large_files_summary_and_limits(tmp_path: Path) -> None:
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    file_path = data_dir / "big.txt"
    line = "a" * 700_000 + "\n"
    file_path.write_text(line * 2, encoding="utf-8")
    original_size = file_path.stat().st_size

    summary = prepare_data.split_large_files(data_dir, max_size_mb=1, max_parts_per_file=3)

    assert summary.total_files == 1
    assert summary.total_parts == 2
    assert summary.total_bytes == original_size


def test_split_large_files_respects_max_parts(tmp_path: Path) -> None:
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    file_path = data_dir / "big.txt"
    line = "b" * 700_000 + "\n"
    file_path.write_text(line * 2, encoding="utf-8")

    with pytest.raises(RuntimeError):
        prepare_data.split_large_files(data_dir, max_size_mb=1, max_parts_per_file=1)
