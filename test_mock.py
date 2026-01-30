import torch
from unittest.mock import patch
import contextlib

def test_compile_mock():
    # Original compile
    print(f"Original torch.compile: {torch.compile}")

    with patch("torch.compile", side_effect=lambda m, *args, **kwargs: "MOCKED"):
        print(f"Mocked torch.compile: {torch.compile}")
        res = torch.compile("model")
        print(f"Result: {res}")
        assert res == "MOCKED"

    print(f"Restored torch.compile: {torch.compile}")
    assert torch.compile != "MOCKED"

if __name__ == "__main__":
    test_compile_mock()
