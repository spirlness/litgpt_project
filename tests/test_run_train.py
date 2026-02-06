from __future__ import annotations

from pathlib import Path

import pytest

import run_train


def test_resolve_resume_value_returns_false_for_disabled_values() -> None:
    for value in [None, "false", "0", "no", False]:
        assert run_train.resolve_resume_value(value) is False


def test_resolve_resume_value_accepts_auto() -> None:
    assert run_train.resolve_resume_value("auto") == "auto"


def test_resolve_resume_value_returns_path(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "lit_model.pth"
    ckpt_path.write_text("ok", encoding="utf-8")

    resolved = run_train.resolve_resume_value(str(ckpt_dir))

    assert resolved == ckpt_path


def test_resolve_resume_value_invalid_raises() -> None:
    with pytest.raises(ValueError):
        run_train.resolve_resume_value("maybe")


def test_apply_train_overrides_is_immutable() -> None:
    train_cfg = {
        "train": {
            "micro_batch_size": 1,
            "global_batch_size": 2,
            "max_tokens": 3,
            "max_seq_length": 4,
            "gradient_checkpointing": True,
        },
        "logger_name": "csv",
    }

    updated = run_train.apply_train_overrides(
        train_cfg,
        micro_batch_size=8,
        global_batch_size=16,
        max_tokens=32,
        max_seq_length=64,
        logger_name="wandb",
        out_dir=Path("out"),
        tokenizer_dir=Path("tok"),
        precision="bf16",
    )

    assert train_cfg["train"]["micro_batch_size"] == 1
    assert train_cfg["logger_name"] == "csv"
    assert updated is not train_cfg
    assert updated["train"] is not train_cfg["train"]
    assert updated["train"]["micro_batch_size"] == 8
    assert updated["logger_name"] == "wandb"
    assert "gradient_checkpointing" not in updated["train"]


def test_apply_model_overrides_casts_norm_eps() -> None:
    model_cfg = {"name": "test", "norm_eps": "1e-5"}

    updated = run_train.apply_model_overrides(model_cfg)

    assert model_cfg["norm_eps"] == "1e-5"
    assert updated["norm_eps"] == pytest.approx(1e-5)
    assert updated is not model_cfg


def test_validate_configs_missing_model_name_raises() -> None:
    model_cfg = {}
    train_cfg = {"train": {}, "data": {"init_args": {}}, "logger_name": "csv"}

    with pytest.raises(RuntimeError):
        run_train.validate_configs(model_cfg, train_cfg)


def test_validate_configs_missing_train_section_raises() -> None:
    model_cfg = {"name": "ok"}
    train_cfg = {"data": {"init_args": {}}, "logger_name": "csv"}

    with pytest.raises(RuntimeError):
        run_train.validate_configs(model_cfg, train_cfg)


def test_validate_configs_missing_data_paths_raises() -> None:
    model_cfg = {"name": "ok"}
    train_cfg = {"train": {}, "data": {"init_args": {}}, "logger_name": "csv"}

    with pytest.raises(RuntimeError):
        run_train.validate_configs(model_cfg, train_cfg)
