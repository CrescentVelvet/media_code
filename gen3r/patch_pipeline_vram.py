#!/usr/bin/env python3
"""[media_code patch] Gen3R pipeline 显存优化（24GB 卡必要）。

两处改动：
1. 编码阶段的 text_encoder（T5，~11GB bf16）与 clip_image_encoder（~2.2GB）在
   prompt/clip 编码后就不再使用，但原版一直占着显存；去噪时 CFG batch=2、
   seq≈32k token，24GB 卡直接溢出到主机内存（实测 shared 21.6GB，单步 >4min）。
   改为编码后搬到 CPU + empty_cache。GEN3R_OFFLOAD_ENCODERS=0 可关闭。
2. 支持 GEN3R_STEPS 覆盖采样步数（默认 50），用于低显存/烟测。

用法: python3 patch_pipeline_vram.py [--revert]
"""
import sys

PATH = "/home/velvet/repos/Gen3R/gen3r/pipeline/pipeline_gen3r.py"

# ── 改动 1：编码后卸载 ─────────────────────────────────────────────
A_ORIG = """        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)"""

A_NEW = """        # [media_code patch] text_encoder / clip_image_encoder 到这一步已用完，
        # 24GB 卡上它们（约 13GB）会和去噪激活抢显存 -> 挪到 CPU。
        if os.environ.get("GEN3R_OFFLOAD_ENCODERS", "1") == "1":
            for _name in ("text_encoder", "clip_image_encoder"):
                _m = getattr(self, _name, None)
                if _m is not None:
                    _m.to("cpu")
            torch.cuda.empty_cache()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)"""

# ── 改动 2：步数覆盖 ──────────────────────────────────────────────
B_ORIG = """        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):"""

B_NEW = """        # [media_code patch] 允许用 GEN3R_STEPS 覆盖步数（低显存烟测）
        _env_steps = os.environ.get("GEN3R_STEPS")
        if _env_steps:
            num_inference_steps = int(_env_steps)

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):"""


def main():
    revert = "--revert" in sys.argv
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    pairs = [(A_ORIG, A_NEW), (B_ORIG, B_NEW)]
    if revert:
        pairs = [(n, o) for o, n in pairs]
    for old, new in pairs:
        if old not in src:
            print("⚠️ 片段不存在：{}".format(old.strip().splitlines()[0][:50]))
            sys.exit(1)
        src = src.replace(old, new, 1)
    with open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("✅ 已{} pipeline 补丁".format("回退" if revert else "应用"))


if __name__ == "__main__":
    main()
