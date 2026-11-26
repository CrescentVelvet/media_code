#!/usr/bin/env python3
"""批量扫描 images 子目录，用官方 ZipSplat 十图重建并输出 zipsplat.ply。

不读取已有相机或 PLY；中间结果保存在 RESULTS_DIR/<scene>/，成品由用户指定的
FINAL_DIR 中的同名场景目录接收。已有成品默认跳过，不覆盖。

WSL 用法：
  DATASET_ROOT=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据 \
  FINAL_DIR=/mnt/d/dataset/测试数据/海思端侧重建拍摄数据 \
  RESULTS_DIR=/mnt/d/output/results/zipsplat_results/hisilicon_10v \
  ONLY=model_man MIN_FREE_MIB=14000 \
  python /mnt/c/code/media_code/zipsplat/batch_hisilicon.py

Env vars:
  DATASET_ROOT, FINAL_DIR, RESULTS_DIR, ONLY, CASE_GLOB, MIN_FREE_MIB,
  WAIT_SECONDS, FORCE, BA, MAX_VIEWS, DRY_RUN
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ["DATASET_ROOT"]).resolve()
FINAL_DIR = Path(os.environ.get("FINAL_DIR", str(ROOT))).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/mnt/d/output/results/zipsplat_results/hisilicon_10v")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_VIEWS = 10
MIN_FREE_MIB = int(os.environ.get("MIN_FREE_MIB", "14000"))
WAIT_SECONDS = int(os.environ.get("WAIT_SECONDS", "0"))
ONLY = {x for x in os.environ.get("ONLY", "").split(",") if x}
CASE_GLOB = os.environ.get("CASE_GLOB", "*")
FORCE = os.environ.get("FORCE", "0") == "1"
RUN_BA = os.environ.get("BA", "1") == "1"
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


def write_json(path, data):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_free_mib():
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return min(int(x.strip()) for x in out.splitlines() if x.strip())


def audit_ply(path):
    """读二进制 PLY 头与全部点数据，避免只凭文件非空验收。"""
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError("missing ply magic")
        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element "):
                fields = text.split()
                in_vertex = fields[1] == "vertex"
                if in_vertex:
                    vertex_count = int(fields[2])
            elif text.startswith("property ") and in_vertex:
                fields = text.split()
                if fields[1] == "list":
                    raise ValueError("vertex list properties are not expected for Gaussian PLY")
                properties.append((fields[1], fields[2]))
            elif text == "end_header":
                break
        if fmt != "binary_little_endian" or not vertex_count:
            raise ValueError(f"unexpected format/count: {fmt}/{vertex_count}")
        sizes = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1, "short": 2,
                 "ushort": 2, "int16": 2, "uint16": 2, "int": 4, "uint": 4,
                 "int32": 4, "uint32": 4, "float": 4, "float32": 4,
                 "double": 8, "float64": 8}
        row_bytes = sum(sizes[t] for t, _ in properties)
        expected = f.tell() + vertex_count * row_bytes
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(f"PLY truncated: actual={actual}, expected={expected}")
    return {"vertex_count": vertex_count, "properties": len(properties),
            "format": fmt, "bytes": actual}


def main():
    if not ROOT.is_dir():
        sys.exit(f"❌ 数据集根目录不存在: {ROOT}")
    if MAX_VIEWS != 10 or os.environ.get("MAX_VIEWS", "10") != "10":
        sys.exit("❌ 本批任务固定十张图像；请用 02_run_inference.sh 跑其它视角数")
    cases = []
    skipped = []
    for scene_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.match(CASE_GLOB)):
        images_dir = scene_dir / "images"
        if not images_dir.is_dir() or (ONLY and scene_dir.name not in ONLY):
            continue
        files = sorted(p for p in images_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGES_SUFFIXES)
        record = {"scene": scene_dir.name, "input_dir": str(images_dir), "image_count": len(files)}
        if len(files) != 10:
            record["status"] = "skipped_invalid_input_count"
            skipped.append(record)
            print(f"⚠️ {scene_dir.name}: {len(files)} 张图像，跳过（要求恰好 10 张）", flush=True)
            continue
        record["output_ply"] = str(FINAL_DIR / scene_dir.name / "zipsplat.ply")
        record["image_names"] = [p.name for p in files]
        cases.append((scene_dir, record))
    if ONLY:
        found = {scene.name for scene, _ in cases}
        missing = ONLY - found - {x["scene"] for x in skipped}
        if missing:
            sys.exit(f"❌ ONLY 未找到有效 images 输入: {sorted(missing)}")
    if not cases:
        print("❌ 没有可运行的十图子目录", flush=True)
    print(f"🚀 扫描完成: {len(cases)} 组十图输入；跳过 {len(skipped)} 组", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "logs").mkdir(exist_ok=True)
    write_json(RESULTS_DIR / "scan.json", {
        "dataset_root": str(ROOT), "final_dir": str(FINAL_DIR),
        "num_cases": len(cases), "cases": [r for _, r in cases], "skipped": skipped})
    if DRY_RUN:
        print("🔍 DRY_RUN=1，仅保存输入清单", flush=True)
        return
    if not FINAL_DIR.is_dir():
        sys.exit(f"❌ 最终输出根目录不存在: {FINAL_DIR}")

    summary = []
    for scene_dir, record in cases:
        name = scene_dir.name
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(exist_ok=True)
        final = FINAL_DIR / name / "zipsplat.ply"
        log_path = RESULTS_DIR / "logs" / f"{name}.log"
        if final.exists() and not FORCE:
            try:
                audit = audit_ply(final)
                metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
                if metrics["num_views"] != 10 or metrics["image_files"] != record["image_names"]:
                    sys.exit(f"❌ 已有成品的输入记录不一致: {final}")
                record.update(status="exists_skipped", audit=audit,
                              elapsed_seconds=sum(metrics["timing_seconds"].values()),
                              returncode=0, log=str(log_path), token_ba=metrics["token_ba"],
                              sha256=sha256(final), result_dir=str(out_dir))
                summary.append(record)
                write_json(RESULTS_DIR / "batch_summary.json", summary)
                write_json(out_dir / "delivery.json", record)
                print(f"⏭️ {name}: 已存在有效 {final}，不覆盖", flush=True)
                continue
            except (OSError, ValueError) as exc:
                sys.exit(f"❌ 已有成品异常，未覆盖: {final}: {exc}")
        if FORCE and final.exists():
            sys.exit(f"❌ 禁止直接覆盖个人数据集中的文件: {final}；请先备份并确认")
        if log_path.exists():
            log_path.rename(log_path.with_suffix(f".previous_{int(time.time())}.log"))

        deadline = time.monotonic() + WAIT_SECONDS
        while gpu_free_mib() < MIN_FREE_MIB and time.monotonic() < deadline:
            print(f"⏳ {name}: 空闲显存 {gpu_free_mib()} MiB < {MIN_FREE_MIB} MiB，30 秒后复查", flush=True)
            time.sleep(30)
        free = gpu_free_mib()
        if free < MIN_FREE_MIB:
            record.update(status="blocked_gpu_memory", gpu_free_mib=free)
            summary.append(record)
            write_json(RESULTS_DIR / "batch_summary.json", summary)
            print(f"⚠️ {name}: 显存不足，保留日志与结果清单，停止后续批次", flush=True)
            break
        print(f"🚀 {name}: 空闲显存 {free} MiB，开始十图前馈 + TokenBA={int(RUN_BA)}", flush=True)
        started = time.time()
        env = os.environ.copy()
        env.update(IMAGES_DIR=str(scene_dir / "images"), OUT_DIR=str(out_dir),
                   SCENE_NAME=name, MAX_VIEWS="10", BA="1" if RUN_BA else "0",
                   SAVE_FEEDFORWARD="1", RENDER_VIDEO="0")
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            proc = subprocess.run(["bash", str(SCRIPT_DIR / "02_run_inference.sh")],
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
        record["elapsed_seconds"] = round(time.time() - started, 3)
        record["returncode"] = proc.returncode
        record["result_dir"] = str(out_dir)
        record["log"] = str(log_path)
        record["token_ba"] = RUN_BA
        if proc.returncode != 0:
            record["status"] = "failed_inference"
            summary.append(record)
            write_json(RESULTS_DIR / "batch_summary.json", summary)
            print(f"❌ {name}: 推理失败 exit={proc.returncode}，日志 {log_path}；停止后续批次", flush=True)
            break
        # 成品使用临时文件名，完整校验通过后才原子改名，避免半成品冒充完成。
        tmp = final.parent / f".zipsplat.{os.getpid()}.tmp.ply"
        if tmp.exists():
            sys.exit(f"❌ 临时输出已存在，需人工检查: {tmp}")
        source = out_dir / "gaussians.ply"
        with source.open("rb") as src, tmp.open("xb") as dst:
            for chunk in iter(lambda: src.read(8 * 2**20), b""):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        record["audit"] = audit_ply(tmp)
        record["sha256"] = sha256(tmp)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics["num_views"] != 10 or metrics["image_files"] != record["image_names"]:
            sys.exit(f"❌ 输入清单与 metrics 不一致: {name}")
        if record["audit"]["vertex_count"] != metrics["num_gaussians"]:
            sys.exit(f"❌ 高斯数量与 metrics 不一致: {name}")
        os.replace(tmp, final)
        record["status"] = "completed"
        record["output_ply"] = str(final)
        summary.append(record)
        write_json(RESULTS_DIR / "batch_summary.json", summary)
        write_json(out_dir / "delivery.json", record)
        print(f"✅ {name}: {final} / {record['audit']['vertex_count']:,} Gaussians / "
              f"{record['elapsed_seconds']:.1f}s", flush=True)
    write_json(RESULTS_DIR / "batch_summary.json", summary)
    done = sum(x["status"] in ("completed", "exists_skipped") for x in summary)
    print(f"🎉 本轮结束: {done}/{len(cases)} 个成品就绪；清单 {RESULTS_DIR / 'batch_summary.json'}", flush=True)
    if done != len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
