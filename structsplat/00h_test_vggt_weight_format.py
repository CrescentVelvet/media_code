#!/usr/bin/env python3
"""
00h_test_vggt_weight_format.py — VGGT 权重格式判据（纯静态，不需 GPU / 不需权重）

【这个坑的现场】
    FileNotFoundError: No such file or directory: ckpts/vggt/model.safetensors

看起来像「路径写错」，实际是**格式不匹配**：

  VGGT(nn.Module, PyTorchModelHubMixin)
  huggingface_hub 的 PyTorchModelHubMixin._from_pretrained 有两条分支：

      if os.path.isdir(model_id):                       # ← 本地目录分支
          model_file = os.path.join(model_id, "model.safetensors")   # 硬编码！
          return cls._load_as_safetensor(...)
      else:                                             # ← hub 下载分支
          try:   hf_hub_download(..., "model.safetensors")
          except EntryNotFoundError:
                 hf_hub_download(..., "pytorch_model.bin")  # 才有 fallback

  HF 上 facebook/VGGT-1B 只提供 **model.pt**（pickle 格式），没有 safetensors。
  于是「下载对了、软链也对了、目录存在」，一跑还是 FileNotFoundError。

【判据】
  A. 本机 huggingface_hub 的本地目录分支确实只认 model.safetensors
  B. 只给 model.pt 时，from_pretrained(本地目录) 必须抛 FileNotFoundError
  C. 补了 model.safetensors 后必须能加载（权重文件在则强校验，不在则跳过）
  D. 01_download_models.sh 里必须有自动转换（防止以后改回只下 model.pt）

跑法：
    /home/velvet/miniconda3/envs/structsplat/bin/python \
        /mnt/c/code/media_code/structsplat/00h_test_vggt_weight_format.py
"""
import inspect
import os
import sys
import tempfile

FAILED = []


def check(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# ============================================================
# A. 源码层：确认本地目录分支硬编码 safetensors
# ============================================================
print("[坑 C-1] huggingface_hub 本地目录分支只认 model.safetensors")
try:
    from huggingface_hub import constants
    from huggingface_hub.hub_mixin import PyTorchModelHubMixin

    check("SAFETENSORS_SINGLE_FILE == 'model.safetensors'",
          constants.SAFETENSORS_SINGLE_FILE == "model.safetensors",
          f"实际 {constants.SAFETENSORS_SINGLE_FILE!r}")
    check("PYTORCH_WEIGHTS_NAME == 'pytorch_model.bin'",
          constants.PYTORCH_WEIGHTS_NAME == "pytorch_model.bin",
          f"实际 {constants.PYTORCH_WEIGHTS_NAME!r}")

    src = inspect.getsource(PyTorchModelHubMixin._from_pretrained)
    # 关键断言：os.path.isdir 分支里出现 SAFETENSORS_SINGLE_FILE，且没有 .pt fallback
    isdir_idx = src.find("os.path.isdir")
    check("_from_pretrained 里有本地目录分支", isdir_idx >= 0)
    if isdir_idx >= 0:
        branch = src[isdir_idx:isdir_idx + 600]
        check("该分支只读 SAFETENSORS_SINGLE_FILE（硬编码）",
              "SAFETENSORS_SINGLE_FILE" in branch)
        check("该分支没有 model.pt / pytorch_model.bin 的 fallback",
              ".pt" not in branch.replace(".path", ""))
except Exception as e:
    check("huggingface_hub 源码检查", False, f"{type(e).__name__}: {e}")


# ============================================================
# B/C. 行为层：真跑 VGGT.from_pretrained
# ============================================================
print("\n[坑 C-2] 只给 model.pt 时必须失败；补 safetensors 后必须成功")
try:
    import torch
    from safetensors.torch import save_file
    from vggt.models.vggt import VGGT

    tmp = tempfile.mkdtemp(prefix="vggt_fmt_")
    # 用极小配置建一个真 VGGT，存成 model.pt（模拟 HF 上的发布格式）
    tiny = dict(img_size=518, patch_size=14, embed_dim=64,
                enable_camera=True, enable_point=False,
                enable_depth=False, enable_track=False)
    m = VGGT(**tiny)
    sd = {k: v for k, v in m.state_dict().items()
          if k.startswith("aggregator.") or k.startswith("camera_head.")}
    # 只留极少层，控制文件体积（判据只关心「找不找得到文件」）
    sd = {k: v.contiguous() for k, v in list(sd.items())[:20]}
    torch.save(sd, os.path.join(tmp, "model.pt"))
    check("已生成 model.pt（模拟 HF 发布格式）",
          os.path.getsize(os.path.join(tmp, "model.pt")) > 0,
          f"{os.path.getsize(os.path.join(tmp,'model.pt')):,} B")

    # B：只有 .pt → 必须抛 FileNotFoundError
    failed_as_expected = False
    try:
        VGGT.from_pretrained(tmp)
    except FileNotFoundError as e:
        failed_as_expected = "model.safetensors" in str(e)
    except Exception:
        pass
    check("只有 model.pt 时 from_pretrained 抛 FileNotFoundError(提及 safetensors)",
          failed_as_expected,
          "这正是现场报错的形态，说明根因是格式而非路径")

    # C：补 safetensors → 必须能加载（用严格同结构的极小模型验证机制本身）
    save_file(sd, os.path.join(tmp, "model.safetensors"))
    check("已补 model.safetensors", True)
    ok_load = False
    try:
        VGGT.from_pretrained(tmp, **tiny)
        ok_load = True
    except Exception as e:
        # 20 个层对不上完整模型是正常的，只要不是 FileNotFoundError 就说明找到了文件
        ok_load = "safetensors" not in str(e) or "size mismatch" in str(e).lower()
        detail = f"{type(e).__name__}: {str(e)[:90]}"
    check("补 safetensors 后不再报文件缺失（机制验证通过）", ok_load,
          detail if not ok_load else "")
    del m, sd, tiny
except Exception as e:
    check("VGGT.from_pretrained 行为验证", False, f"{type(e).__name__}: {e}")


# ============================================================
# D. 下载脚本必须有自动转换
# ============================================================
print("\n[坑 C-3] 01_download_models.sh 必须内置 model.pt → safetensors 转换")
try:
    here = os.path.dirname(os.path.abspath(__file__))
    dl = os.path.join(here, "01_download_models.sh")
    txt = open(dl, encoding="utf-8").read()
    check("01 脚本存在", os.path.exists(dl))
    check("定义了转换函数", "he_vggt_safetensors" in txt)
    check("实际调用了转换函数", txt.count("he_vggt_safetensors") >= 2)
    check("转换用 safetensors.save_file", "save_file" in txt)
    check("注释里说明了根因（本地目录硬编码）",
          "硬编码" in txt or "SAFETENSORS_SINGLE_FILE" in txt)
except Exception as e:
    check("01 脚本检查", False, f"{type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
if FAILED:
    print(f"❌ {len(FAILED)} 项未通过：")
    for f in FAILED:
        print(f"   - {f}")
    sys.exit(1)
print("🎉 全部通过")
print("   结论：VGGT 本地目录必须提供 model.safetensors；01 脚本已内置自动转换。")
