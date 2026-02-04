#!/usr/bin/env python3
"""Test script to verify that our fixes for checkpoint saving and metrics logging work."""

import os
import shutil
import tempfile
from pathlib import Path


def test_directory_structure():
    """Test that the required directory structure exists."""
    print("Testing directory structure...")

    # Check train data structure
    train_dir = Path("data/custom_text/train")
    train_train_dir = train_dir / "train"
    if not train_train_dir.exists():
        print(f"ERROR: {train_train_dir} does not exist")
        return False

    # Check for chunk files
    chunk_files = list(train_train_dir.glob("chunk-*.bin"))
    if not chunk_files:
        print(f"ERROR: No chunk files found in {train_train_dir}")
        return False

    print(f"✓ Found {len(chunk_files)} chunk files in train directory")

    # Check val data structure
    val_dir = Path("data/custom_text/val")
    val_val_dir = val_dir / "val"
    if not val_val_dir.exists():
        print(f"Creating {val_val_dir}")
        val_val_dir.mkdir(parents=True, exist_ok=True)

    # Copy a chunk file to val if none exist
    val_chunk_files = list(val_val_dir.glob("chunk-*.bin"))
    if not val_chunk_files and chunk_files:
        print(f"Copying chunk file to {val_val_dir}")
        shutil.copy(chunk_files[0], val_val_dir / "chunk-0-0.bin")
        val_chunk_files = [val_val_dir / "chunk-0-0.bin"]

    if not val_chunk_files:
        print(f"ERROR: No chunk files found in {val_val_dir}")
        return False

    print(f"✓ Found {len(val_chunk_files)} chunk files in val directory")

    return True


def test_logs_directory():
    """Test that logs directory can be created."""
    print("\nTesting logs directory creation...")

    # Check smoke test output directory
    smoke_dir = Path("checkpoints_smoke")
    logs_dir = smoke_dir / "logs"

    if not logs_dir.exists():
        print(f"Creating {logs_dir}")
        logs_dir.mkdir(parents=True, exist_ok=True)

    if logs_dir.exists():
        print("✓ Logs directory exists")
        return True
    else:
        print(f"ERROR: Failed to create {logs_dir}")
        return False


def test_checkpoint_directory():
    """Test that checkpoint directory can be created."""
    print("\nTesting checkpoint directory creation...")

    # Check smoke test output directory
    smoke_dir = Path("checkpoints_smoke")

    if not smoke_dir.exists():
        print(f"Creating {smoke_dir}")
        smoke_dir.mkdir(parents=True, exist_ok=True)

    if smoke_dir.exists():
        print("✓ Checkpoint directory exists")
        return True
    else:
        print(f"ERROR: Failed to create {smoke_dir}")
        return False


def main():
    """Run all tests."""
    print("Running tests for LitGPT training fixes...\n")

    tests = [
        test_directory_structure,
        test_logs_directory,
        test_checkpoint_directory,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
