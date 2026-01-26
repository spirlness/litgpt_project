from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def dir_fingerprint(path: Path) -> dict:
    file_count = 0
    total_size = 0
    sha = hashlib.sha256()

    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        file_count += 1
        try:
            stat = file_path.stat()
            total_size += stat.st_size
        except OSError:
            continue

        sha.update(str(file_path.relative_to(path)).encode("utf-8", errors="ignore"))
        sha.update(b"\0")
        sha.update(str(stat.st_size).encode("ascii"))
        sha.update(b"\0")
        sha.update(str(int(stat.st_mtime)).encode("ascii"))
        sha.update(b"\0")

    return {
        "file_count": file_count,
        "total_size_bytes": total_size,
        "fingerprint_sha256": sha.hexdigest(),
    }


def log_dataset_to_wandb(
    *,
    data_dir: Path,
    project: str,
    entity: str | None,
    artifact_name: str,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    run_name: str | None = None,
    artifact_root_name: str = "custom_text",
) -> None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("wandb is required to log dataset artifacts") from exc

    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    fingerprint = dir_fingerprint(data_dir)
    metadata = {
        "data_dir": str(data_dir),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        **fingerprint,
    }

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="data-prep",
        name=run_name,
        tags=tags or None,
        config={"dataset": metadata},
        reinit="finish_previous",
    )

    artifact = wandb.Artifact(name=artifact_name, type="dataset", metadata=metadata)
    artifact.add_dir(str(data_dir), name=artifact_root_name)

    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "included_root": artifact_root_name,
                    "fingerprint": fingerprint,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifact.add_file(str(manifest_path), name="manifest.json")

    logged = run.log_artifact(artifact, aliases=aliases or None)
    mode = getattr(getattr(run, "settings", None), "mode", None)
    if mode != "offline":
        try:
            if hasattr(logged, "wait"):
                logged.wait()
            else:
                artifact.wait()
        except Exception:
            # Some modes/backends do not support wait.
            pass
    run.finish()


def _main() -> int:
    parser = argparse.ArgumentParser(description="Upload a local dataset directory as a W&B Artifact")
    parser.add_argument("--data-dir", type=Path, default=Path("data/custom_text"))
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT"),
    )
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-artifact",
        type=str,
        default=os.environ.get("WANDB_DATA_ARTIFACT", "dataset-custom_text"),
    )
    parser.add_argument(
        "--wandb-alias",
        action="append",
        default=os.environ.get("WANDB_DATA_ALIASES", "latest").split(",") if os.environ.get("WANDB_DATA_ALIASES") else ["latest"],
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else [],
    )
    parser.add_argument("--wandb-run-name", type=str, default=os.environ.get("WANDB_RUN_NAME"))
    args = parser.parse_args()

    if not args.wandb_project:
        raise SystemExit(
            "Missing W&B project. Provide --wandb-project <project> or set $env:WANDB_PROJECT='<project>'."
        )

    log_dataset_to_wandb(
        data_dir=args.data_dir,
        project=args.wandb_project,
        entity=args.wandb_entity,
        artifact_name=args.wandb_artifact,
        aliases=[a for a in args.wandb_alias if a],
        tags=[t for t in args.wandb_tag if t],
        run_name=args.wandb_run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
