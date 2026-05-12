import time
import torch

assert torch.cuda.is_available(), "No GPU found"
print(f"GPU: {torch.cuda.get_device_name(0)}")

a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")

for precision in ["highest", "medium"]:
    torch.set_float32_matmul_precision(precision)
    for _ in range(10):  # warmup
        torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(50):
        torch.mm(a, b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / 50 * 1000

    print(f"  {precision:>8s}: {ms:.2f} ms")
