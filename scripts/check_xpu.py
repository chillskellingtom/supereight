import torch

print("torch:", torch.__version__)
print("xpu available:", hasattr(torch, "xpu") and torch.xpu.is_available())
if hasattr(torch, "xpu"):
    print("xpu count:", torch.xpu.device_count())
    for i in range(torch.xpu.device_count()):
        print(i, torch.xpu.get_device_name(i))

