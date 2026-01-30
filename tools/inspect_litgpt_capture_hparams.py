import inspect

from litgpt.utils import capture_hparams

print("capture_hparams:", capture_hparams)
print("module:", capture_hparams.__module__)

try:
    src = inspect.getsource(capture_hparams)
    print("\n--- capture_hparams source (head) ---")
    print(src[:4000])
except Exception as exc:
    print("getsource failed:", exc)
