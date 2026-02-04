#!/usr/bin/env python3
"""
Final verification script to confirm that checkpoint saving and logging fixes work.
This script tests the core functionality without going through the full training pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path
import torch
from unittest.mock import Mock, MagicMock


def test_checkpoint_save_function():
    """Test that the save_checkpoint function works correctly."""
    print("Testing checkpoint save function...")

    try:
        # Import the actual function
        sys.path.insert(0, "/home/lee/litgpt_project/.venv/lib/python3.12/site-packages")
        from litgpt.pretrain import save_checkpoint

        # Create mocks
        fabric = Mock()
        fabric.global_rank = 0
        fabric.save = Mock()
        fabric.print = Mock()

        model = Mock()
        model.config = Mock()
        state = {"model": model}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint" / "lit_model.pth"

            # Call the function
            save_checkpoint(fabric, state, None, checkpoint_path)

            # Verify it was called correctly
            fabric.save.assert_called_once_with(checkpoint_path, state)
            fabric.print.assert_called_once()

        print("✅ Checkpoint save function works correctly")
        return True

    except Exception as e:
        print(f"❌ Checkpoint save function test failed: {e}")
        return False


def test_csv_logger_setup():
    """Test that CSV logger setup works correctly."""
    print("\nTesting CSV logger setup...")

    try:
        # Import the actual function
        sys.path.insert(0, "/home/lee/litgpt_project/.venv/lib/python3.12/site-packages")
        from litgpt.utils import choose_logger
        from lightning.pytorch.loggers import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Create the logger
            logger = choose_logger(logger_name="csv", out_dir=out_dir, name="test_run", log_interval=1)

            # Verify it's the correct type
            if isinstance(logger, CSVLogger):
                print("✅ CSV logger setup works correctly")
                return True
            else:
                print(f"❌ Expected CSVLogger, got {type(logger)}")
                return False

    except Exception as e:
        print(f"❌ CSV logger setup test failed: {e}")
        return False


def test_logs_directory_creation():
    """Test that logs directory is created properly."""
    print("\nTesting logs directory creation...")

    try:
        # Import the actual function
        sys.path.insert(0, "/home/lee/litgpt_project/.venv/lib/python3.12/site-packages")
        from litgpt.utils import choose_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            logs_dir = out_dir / "logs" / "csv"

            # Create the logger (this should create the directory)
            logger = choose_logger(logger_name="csv", out_dir=out_dir, name="test_run", log_interval=1)

            # Check if directory was created
            if logs_dir.exists():
                print("✅ Logs directory creation works correctly")
                return True
            else:
                print("❌ Logs directory was not created")
                return False

    except Exception as e:
        print(f"❌ Logs directory creation test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("Running final verification of fixes...\n")

    tests = [
        test_checkpoint_save_function,
        test_csv_logger_setup,
        test_logs_directory_creation,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\nFinal Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All core functionality fixes verified successfully!")
        print("\nSummary of verified fixes:")
        print("✅ Checkpoint saving function works")
        print("✅ CSV logger creation works")
        print("✅ Logs directory creation works")
        return True
    else:
        print("❌ Some fixes could not be verified")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
