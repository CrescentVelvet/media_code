#!/usr/bin/env python3
"""给 vendored diffusers 打一个离线模式补丁（幂等）。

为什么需要：
  `WanTransformer3DModel.from_pretrained(...)` 在 `local_files_only` 为 None 时，
  分片权重（`diffusion_pytorch_model.safetensors.index.json`）会走
  `utils/hub_utils.py::_get_checkpoint_shard_files` 的「Hub 模型 ID」分支，
  里面**无条件调用** `model_info()`（一次联网 API 请求）。
  本项目权重全在本地 cache，且 `02_run_inference.sh` 会导出 `HF_HUB_OFFLINE=1`，
  该请求直接被 huggingface_hub 拦成 `OfflineModeIsEnabled`，加载阶段就挂。

  而 `modeling_utils.py` 里 `local_files_only = kwargs.pop("local_files_only", None)`
  默认是 `None`（不是 `HF_HUB_OFFLINE`），所以必须显式兜底。

补丁内容：`local_files_only` 为 None 时，回退读环境变量 `HF_HUB_OFFLINE`。
只改动 vendored 依赖（`$FLASHWORLD_DEPS/diffusers`），**不碰官方 FlashWorld 代码**。

Env vars:
  FLASHWORLD_DEPS  vendored 依赖根目录（默认 ~/repos/_flashworld_deps）
"""
import os
import re
import sys
from pathlib import Path

DEPS = Path(os.environ.get("FLASHWORLD_DEPS", Path.home() / "repos" / "_flashworld_deps"))
TARGET = DEPS / "diffusers" / "src" / "diffusers" / "models" / "modeling_utils.py"

OLD = '        local_files_only = kwargs.pop("local_files_only", None)\n'
NEW = (
    '        local_files_only = kwargs.pop("local_files_only", None)\n'
    '        # [flashworld-patch] 见 flashworld/00b_patch_deps.py：\n'
    '        # 分片 ckpt 会走 model_info() 联网，HF_HUB_OFFLINE=1 时直接抛 OfflineModeIsEnabled\n'
    '        if local_files_only is None:\n'
    '            local_files_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")\n'
)
MARKER = "[flashworld-patch]"


def main():
    if not TARGET.is_file():
        sys.exit("❌ 找不到 %s（先跑 00a_setup_env.sh）" % TARGET)

    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("⏭️  已打过补丁，跳过: %s" % TARGET)
        return

    # 只替换 first 处的首行（modeling_utils 里有两处同名赋值，第二处在别的函数，不必动）
    idx = src.find(OLD)
    if idx < 0:
        sys.exit("❌ 未找到锚点行，diffusers 版本可能已变，需人工核对: %s" % TARGET)
    src = src[:idx] + NEW + src[idx + len(OLD):]

    if not re.search(r"^import os$|^import os\b", src, re.M):
        sys.exit("❌ modeling_utils.py 里没有 import os，补丁不适用")

    TARGET.write_text(src, encoding="utf-8", newline="\n")
    print("✅ 已打补丁: %s" % TARGET)
    print("   行为：HF_HUB_OFFLINE=1 时 from_pretrained 默认 local_files_only=True")


if __name__ == "__main__":
    main()
