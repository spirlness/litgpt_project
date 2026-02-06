import torch
import time


@torch.compile
def fn(x, y):
    return x + y * x


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(0)
    x = torch.randn(1024, 1024).cuda()
    y = torch.randn(1024, 1024).cuda()
    print("Compiling...")
    start = time.time()
    fn(x, y)
    print(f"Compiled in {time.time() - start:.2f}s")

    start = time.time()
    for _ in range(100):
        fn(x, y)
    print(f"100 iterations in {time.time() - start:.4f}s")
