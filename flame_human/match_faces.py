#!/usr/bin/env python3
"""match_faces.py — 阶段二：face bbox ↔ person track ID 关联（三层判据）。

为什么不用字面 IoU(face_box, person_box)：
人脸框基本完全包含在人体框内，IoU = |face|/|person|，量级只有 0.02~0.1；
两人并排/遮挡时 A 的脸框会同时落在 A、B 的人体框内，两个 IoU 只差分母
|person|，区分度趋近于零 —— 会得到看起来合理、实际串号的 pid。

三层判据：
  1. 主判据 point-in-mask：人脸框中心是否落在 person mask 内
  2. 次判据 IoM = |face ∩ mask| / |face|（分母用 face 面积，天然处理包含关系）
  3. 全局分配：逐帧匈牙利一对一（贪心在 3 人时可能陷入次优绑定）
     + 时序约束（允许缺帧，不允许 ID 跳变）

漏检帧策略：不兜底。该帧该 person 静默跳过，靠下游的成功率阈值降级
（见设计文档 §2）。

Env:
  FACE_BOXES     阶段一输出（默认 $RESULTS_DIR/02_faces/face_boxes.json）
  MASKS_DIR      person mask 目录，文件名 {stem}.p{oid}.mask.png
                 （默认 $PERSON_MASKS_DIR）
  IMAGES_DIR     图像目录（取画幅尺寸；默认 $SOURCE_DIR/images）
  OUT_JSON       输出（默认 $RESULTS_DIR/03_match/face_person_match.json）
  INSIDE_W       point-in-mask 权重（默认 2.0，要压过 IoM）
  MIN_IOM        接受匹配的最小 IoM（默认 0.3）
  DROP_SUSPECT   时序跳变的匹配是否直接丢弃（默认 0，只标记）
  MAX_JUMP_PX    相邻两帧同一人面部中心的最大位移/帧（默认 200）
"""
import os
import sys
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from scipy.optimize import linear_sum_assignment

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def log(m):
    print(m, flush=True)


def load_person_masks(masks_dir):
    """→ {oid: {stem: Path}}。文件名约定 {stem}.p{oid}.mask.png。"""
    out = {}
    if not Path(masks_dir).is_dir():
        sys.exit(f"❌ person mask 目录不存在: {masks_dir}")
    for f in sorted(Path(masks_dir).iterdir()):
        if not f.name.endswith(".mask.png") or ".p" not in f.name:
            continue
        stem, rest = f.name.split(".p", 1)
        m = re.match(r"(\d+)", rest)
        if not m:
            continue
        out.setdefault(int(m.group(1)), {})[stem] = f
    return out


def read_mask(path, size):
    """读 mask 并 resize 到图像尺寸（nearest，保持 0/1）。"""
    m = np.array(Image.open(path).convert("L"))
    if m.shape[1] != size[0] or m.shape[0] != size[1]:
        m = np.array(Image.fromarray(m).resize(size, Image.NEAREST))
    return m > 127


def main():
    source_dir = os.environ.get("SOURCE_DIR", "")
    results_dir = os.environ.get("RESULTS_DIR", "")
    face_boxes = Path(os.environ.get(
        "FACE_BOXES", f"{results_dir}/02_faces/face_boxes.json"))
    masks_dir = os.environ.get(
        "MASKS_DIR", os.environ.get("PERSON_MASKS_DIR", ""))
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    out_json = Path(os.environ.get(
        "OUT_JSON", f"{results_dir}/03_match/face_person_match.json"))
    inside_w = float(os.environ.get("INSIDE_W", "2.0"))
    min_iom = float(os.environ.get("MIN_IOM", "0.3"))
    drop_suspect = int(os.environ.get("DROP_SUSPECT", "0"))
    max_jump = float(os.environ.get("MAX_JUMP_PX", "200"))

    if not face_boxes.exists():
        sys.exit(f"❌ 缺少阶段一输出: {face_boxes}")

    faces = json.loads(face_boxes.read_text())
    persons = load_person_masks(masks_dir)
    oids = sorted(persons)
    log(f"🔗 [阶段二] face ↔ person 关联")
    log(f"  👥 person tracks: {oids}")
    log(f"  🖼️  {len(faces['frames'])} 帧待处理")

    out_frames = {}
    warnings = []
    # oid → [(frame_no, cx, cy, stem)]，用于时序连续性检查
    traj = {o: [] for o in oids}

    frame_list = sorted(faces["frames"].keys())
    for fi, stem in enumerate(frame_list):
        rec = faces["frames"][stem]
        W, H = rec.get("W", 0), rec.get("H", 0)
        if not W or not H:
            # 取不到画幅就找图像文件读一次尺寸
            for ext in IMG_EXTS:
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    W, H = Image.open(p).size
                    break
        if not W or not H:
            warnings.append(f"{stem}: 无画幅尺寸，跳过")
            out_frames[stem] = {"persons": {}}
            continue

        fboxes = rec.get("faces", [])
        # 本帧有 mask 的 person
        cur = [o for o in oids if stem in persons[o]]
        rec_out = {"persons": {}}
        if not fboxes or not cur:
            out_frames[stem] = rec_out
            continue

        # 预读本帧所有 person mask
        masks = {o: read_mask(persons[o][stem], (W, H)) for o in cur}

        # 代价矩阵 (n_faces × n_persons)
        n_f, n_p = len(fboxes), len(cur)
        cost = np.zeros((n_f, n_p), dtype=np.float64)
        iom_m = np.zeros((n_f, n_p), dtype=np.float64)
        ins_m = np.zeros((n_f, n_p), dtype=np.int8)
        for a, fb in enumerate(fboxes):
            x0, y0, x1, y1 = fb["bbox"]
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
            if x1 <= x0 or y1 <= y0:
                cost[a, :] = 1e3
                continue
            cx = int(np.clip((x0 + x1) * 0.5, 0, W - 1))
            cy = int(np.clip((y0 + y1) * 0.5, 0, H - 1))
            area = float((x1 - x0) * (y1 - y0))
            for b, o in enumerate(cur):
                mk = masks[o]
                inter = float(mk[y0:y1, x0:x1].sum())
                iom = inter / max(area, 1.0)
                inside = 1 if mk[cy, cx] else 0
                iom_m[a, b] = iom
                ins_m[a, b] = inside
                # 代价取负：inside 权重压过 iom，保证主判据优先
                cost[a, b] = -(inside_w * inside + iom)

        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            o = cur[b]
            if iom_m[a, b] < min_iom:
                warnings.append(
                    f"{stem}: person {o} 的最佳匹配 IoM={iom_m[a,b]:.3f} "
                    f"< {min_iom}，丢弃")
                continue
            fb = fboxes[a]
            x0, y0, x1, y1 = fb["bbox"]
            rec_out["persons"][str(o)] = {
                "face_idx": int(a),
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "iom": float(iom_m[a, b]),
                "inside": int(ins_m[a, b]),
                "score": float(fb.get("score", 0.0)),
                "kps": fb.get("kps", []),
            }
            traj[o].append((fi, (x0 + x1) * 0.5, (y0 + y1) * 0.5, stem))
        out_frames[stem] = rec_out

    # ── 时序连续性检查：允许缺帧，不允许 ID 跳变 ────────────────────────
    n_suspect = 0
    for o in oids:
        t = sorted(traj[o])
        for k in range(1, len(t)):
            f0, x0_, y0_, s0 = t[k - 1]
            f1, x1_, y1_, s1 = t[k]
            gap = max(f1 - f0, 1)
            d = float(np.hypot(x1_ - x0_, y1_ - y0_))
            if d > max_jump * gap:
                n_suspect += 1
                msg = (f"person {o}: {s0}→{s1} 面部中心位移 {d:.0f}px "
                       f"(>{max_jump:.0f}px/帧 × {gap} 帧)，疑似 ID 跳变")
                warnings.append(msg)
                if drop_suspect:
                    if s1 in out_frames and str(o) in out_frames[s1]["persons"]:
                        del out_frames[s1]["persons"][str(o)]
                        warnings.append(f"  → DROP_SUSPECT=1，已丢弃 {s1}")

    # ── 汇总 ────────────────────────────────────────────────────────────
    persons_stat = {}
    for o in oids:
        ioms = []
        for stem, r in out_frames.items():
            if str(o) in r["persons"]:
                ioms.append(r["persons"][str(o)]["iom"])
        persons_stat[str(o)] = {
            "n_frames": len(ioms),
            "mean_iom": float(np.mean(ioms)) if ioms else 0.0,
            "coverage": len(ioms) / max(len(frame_list), 1),
        }
        log(f"  p{o:02d}: 匹配 {len(ioms)}/{len(frame_list)} 帧 "
            f"({persons_stat[str(o)]['coverage']*100:.0f}%), "
            f"mean IoM {persons_stat[str(o)]['mean_iom']:.3f}")

    out = {
        "meta": {
            "face_boxes": str(face_boxes),
            "masks_dir": str(masks_dir),
            "inside_w": inside_w,
            "min_iom": min_iom,
            "max_jump_px": max_jump,
            "drop_suspect": bool(drop_suspect),
            "n_frames": len(frame_list),
            "n_persons": len(oids),
        },
        "persons": persons_stat,
        "frames": out_frames,
        "warnings": warnings[:200],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1))
    log(f"💾 {out_json}")
    if n_suspect:
        log(f"  ⚠️ {n_suspect} 处疑似 ID 跳变（详见 warnings；"
            f"DROP_SUSPECT=1 可直接丢弃）")
    log("🎉 关联完成")


if __name__ == "__main__":
    main()
