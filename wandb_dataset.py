"""
Utility for logging datasets to Weights & Biases as Artifacts.

This module provides functionality to upload local dataset directories
to W&B Artifacts, enabling versioning and sharing of training data.
"""

import os
from pathlib import Path
from typing import List, Optional

import wandb


def log_dataset_to_wandb(
    data_dir: Path,
    project: str,
    entity: Optional[str] = None,
    artifact_name: str = "dataset",
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    run_name: Optional[str] = None,
    artifact_type: str = "dataset",
) -> wandb.Artifact:
    """
    Log a dataset directory to Weights & Biases as an Artifact.

    Args:
        data_dir: Path to the dataset directory to upload
        project: W&B project name
        entity: W&B entity (username or team). If None, uses default entity
        artifact_name: Name for the artifact
        aliases: List of aliases for this artifact version (e.g., ["latest", "v1"])
        tags: List of tags for the wandb run
        run_name: Name for the wandb run
        artifact_type: Type of artifact (default: "dataset")

    Returns:
        wandb.Artifact: The created artifact object

    Raises:
        FileNotFoundError: If data_dir does not exist
        RuntimeError: If wandb initialization or upload fails
    """
    # Validate data directory
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not data_dir.is_dir():
        data_dir = data_dir.parent

    # Initialize W&B run
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type="dataset_upload",
        tags=tags or [],
        settings={"silent": True},
    )

    try:
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata={
                "data_dir": str(data_dir),
                "num_files": sum(1 for _ in data_dir.rglob("*") if _.is_file()),
            },
        )

        # Add entire directory to artifact
        print(f"Adding directory '{data_dir}' to artifact...")
        artifact.add_dir(str(data_dir), name=data_dir.name)

        # Log and save artifact
        print(f"Logging artifact '{artifact_name}'...")
        run.log_artifact(artifact, aliases=aliases or [])

        # Wait for upload to complete
        artifact.wait()

        print(f"Artifact uploaded successfully: {artifact.name}")
        print(f"Artifact ID: {artifact.id}")

        return artifact

    except Exception as e:
        raise RuntimeError(f"Failed to upload dataset to W&B: {e}") from e

    finally:
        # Close the run
        run.finish()


if __name__ == "__main__":
    # CLI interface for standalone usage
    import argparse

    parser = argparse.ArgumentParser(description="Upload dataset directory to W&B as an Artifact")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory to upload",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "my-project"),
        help="W&B project name (default: $WANDB_PROJECT)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (default: $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default="dataset",
        help="Artifact name (default: dataset)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        action="append",
        default=["latest"],
        help="Artifact alias (can be specified multiple times, default: latest)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        action="append",
        default=[],
        help="Run tag (can be specified multiple times)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    args = parser.parse_args()

    log_dataset_to_wandb(
        data_dir=args.data_dir,
        project=args.project,
        entity=args.entity,
        artifact_name=args.artifact_name,
        aliases=args.alias,
        tags=args.tag,
        run_name=args.run_name,
    )
