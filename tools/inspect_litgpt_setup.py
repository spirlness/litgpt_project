import inspect
import litgpt.pretrain as p

print("setup:", p.setup)
print("module:", p.setup.__module__)

try:
    src = inspect.getsource(p.setup)
    print("\n--- setup source (head) ---")
    print(src[:4000])
except Exception as exc:
    print("getsource failed:", exc)
