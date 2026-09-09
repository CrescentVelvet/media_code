#!/usr/bin/env python3
"""批量验证 flame_human env 的包 import 状态（00a 装完后自检用）。"""
import importlib

MODULES = [
    "numpy", "torch", "torchvision", "cv2", "mediapipe", "smplx", "chumpy",
    "scipy", "plyfile", "skimage", "sklearn", "trimesh", "matplotlib",
    "yaml", "tqdm", "PIL",
]

fails = 0
for m in MODULES:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "")
        print(f"  OK   {m} {ver}")
    except Exception as e:
        fails += 1
        print(f"  FAIL {m}: {type(e).__name__}: {e}")

# torch <-> numpy 互操作（numpy 2.x 会挂的那个）
try:
    import torch
    t = torch.tensor([1.0, 2.0])
    assert t.numpy().tolist() == [1.0, 2.0]
    import numpy as np
    a = np.ones((2, 2), dtype=np.float32)
    assert torch.from_numpy(a).sum().item() == 4.0
    print(f"  OK   torch<->numpy interop (numpy {np.__version__})")
except Exception as e:
    fails += 1
    print(f"  FAIL torch<->numpy: {e}")

# CUDA 可用性（3090）
try:
    import torch
    print(f"  OK   cuda available: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'})")
except Exception as e:
    print(f"  WARN cuda check: {e}")

print("RESULT:", "PASS" if fails == 0 else f"{fails} FAIL")
