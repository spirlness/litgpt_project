import importlib
import os

REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "torchaudio",
    "litgpt",
    "lightning",
    "pytorch_lightning",
    "tokenizers",
    "wandb",
    "safetensors",
]

OPTIONAL_MODULES = [
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
]

if os.environ.get("SKIP_BNB_RUNTIME") != "1":
    OPTIONAL_MODULES.append("bitsandbytes")


def try_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def main() -> int:
    failures: list[tuple[str, str]] = []
    optional_failures: list[tuple[str, str]] = []
    for name in REQUIRED_MODULES:
        ok, err = try_import(name)
        if ok:
            print("OK ", name)
        else:
            print("FAIL", name, err)
            failures.append((name, err))

    for name in OPTIONAL_MODULES:
        ok, err = try_import(name)
        if ok:
            print("OK ", name)
        else:
            print("WARN", name, err)
            optional_failures.append((name, err))

    import torch  # noqa: E402

    print("\nTORCH:", torch.__version__)
    print("CUDA version string:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    # Minimal CUDA runtime check (catches driver/runtime mismatch and some ABI issues)
    if torch.cuda.is_available():
        x = torch.randn((256, 256), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("CUDA matmul mean:", float(y.mean().item()))

    # bitsandbytes: import alone isn't always enough on Windows; do a tiny touch if possible.
    if os.environ.get("SKIP_BNB_RUNTIME") == "1":
        print("\nBNB: skipped (SKIP_BNB_RUNTIME=1)")
    else:
        try:
            import bitsandbytes as bnb  # noqa: E402

            print("\nBNB:", getattr(bnb, "__version__", "<unknown>"))
            # Some builds expose functional API; if present, touch it.
            fn = getattr(bnb, "functional", None)
            if fn is not None:
                print("BNB functional:", "present")
        except Exception as exc:  # noqa: BLE001
            print("\nBNB_RUNTIME_FAIL:", repr(exc))
            # On Windows without CUDA, bitsandbytes might fail to load libs.
            # Treat as warning if CUDA is not available.
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False

            if cuda_available:
                failures.append(("bitsandbytes(runtime)", repr(exc)))
            else:
                 optional_failures.append(("bitsandbytes(runtime)", repr(exc)))

    if failures:
        print("\nFAILED_IMPORTS_OR_RUNTIME:")
        for name, err in failures:
            print("-", name, err)
        if optional_failures:
            print("\nOPTIONAL_IMPORT_WARNINGS:")
            for name, err in optional_failures:
                print("-", name, err)
        return 2

    if optional_failures:
        print("\nOPTIONAL_IMPORT_WARNINGS:")
        for name, err in optional_failures:
            print("-", name, err)

    print("\nALL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
