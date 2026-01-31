import torch
import torch.nn.functional as F


def verify_flash_attention():
    if not torch.cuda.is_available():
        print("CUDA not available. Flash Attention requires a GPU.")
        return

    # Flash Attention 2 requires Ampere (8.0+) or newer
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU Compute Capability: {major}.{minor}")

    # Check if flash_attn package is installed
    try:
        import flash_attn

        print(f"flash_attn package version: {flash_attn.__version__}")
    except ImportError:
        print("flash_attn package not found.")

    # Test SDPA with Flash Attention backend explicitly
    B, H, T, D = 2, 8, 1024, 64
    dtype = torch.bfloat16  # Flash Attention prefers bf16 or f16
    q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)

    # Check available backends in torch
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        print("\nSupported SDPA Backends in this PyTorch version:")
        # Test if FLASH_ATTENTION is available
        q_test = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            try:
                F.scaled_dot_product_attention(q_test, q_test, q_test)
                print("  - FLASH_ATTENTION: Available and Working")
            except Exception as e:
                print(f"  - FLASH_ATTENTION: Not available for this config ({e})")

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            print("  - EFFICIENT_ATTENTION: Available")
        with sdpa_kernel(SDPBackend.MATH):
            print("  - MATH: Available")
    except ImportError:
        print("torch.nn.attention.SDPBackend not found (older torch).")


if __name__ == "__main__":
    verify_flash_attention()
