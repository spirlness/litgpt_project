#!/usr/bin/env python3
"""Direct test of checkpoint saving and logging functionality."""

import os
import sys
import tempfile
from pathlib import Path
import torch
import yaml
from unittest.mock import Mock, patch

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))


def test_checkpoint_saving():
    """Test that checkpoint saving function works correctly."""
    print("Testing checkpoint saving...")

    try:
        # Import the save_checkpoint function
        from litgpt.pretrain import save_checkpoint

        # Create a mock fabric
        fabric = Mock()
        fabric.global_rank = 0
        fabric.save = Mock()

        # Create a mock state
        model = Mock()
        model.config = Mock()
        state = {"model": model}

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint" / "lit_model.pth"

            # Test the save_checkpoint function
            save_checkpoint(fabric, state, None, checkpoint_path)

            # Check that fabric.save was called
            fabric.save.assert_called_once_with(checkpoint_path, state)
            print("✓ save_checkpoint function works correctly")
            return True

    except Exception as e:
        print(f"✗ save_checkpoint test failed: {e}")
        return False


def test_csv_logger_creation():
    """Test that CSV logger can be created correctly."""
    print("\nTesting CSV logger creation...")

    try:
        # Import the choose_logger function
        from litgpt.utils import choose_logger
        from lightning.pytorch.loggers import CSVLogger

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Test creating CSV logger
            logger = choose_logger(logger_name="csv", out_dir=out_dir, name="test", log_interval=1)

            # Check that we got a CSVLogger instance
            if isinstance(logger, CSVLogger):
                print("✓ CSV logger created successfully")
                return True
            else:
                print(f"✗ Expected CSVLogger, got {type(logger)}")
                return False

    except Exception as e:
        print(f"✗ CSV logger test failed: {e}")
        return False


def test_logs_directory_creation():
    """Test that logs directory is created correctly."""
    print("\nTesting logs directory creation...")

    try:
        # Import the choose_logger function
        from litgpt.utils import choose_logger

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            logs_dir = out_dir / "logs"

            # Test creating CSV logger
            logger = choose_logger(logger_name="csv", out_dir=out_dir, name="test", log_interval=1)

            # Check that logs directory was created
            if logs_dir.exists():
                print("✓ Logs directory created successfully")
                return True
            else:
                print("✗ Logs directory was not created")
                return False

    except Exception as e:
        print(f"✗ Logs directory test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running tests for checkpoint saving and logging...\n")

    tests = [
        test_checkpoint_saving,
        test_csv_logger_creation,
        test_logs_directory_creation,
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
