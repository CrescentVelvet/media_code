#!/usr/bin/env python3
"""fig12/13：方法对照示意图（PPT 整页，按 figkit/SPEC.md 出图，1920×1080）。

fig12  DA3-GS         vs  InfiniSplat v2
fig13  InfiniSplat v2 vs  v3（SplatAnything）

版式：左右两栏各 6 步流水线（同 y 逐行对齐，第 6 步差异一眼可见）+ 下方 5 行差异表。
口径：机制来自代码/文档（splatanything/NOTES.md、infinisplat_v2/README_wsl.md、
compare_4methods/FINDINGS.md §2）；耗时为 3090 实测，不跨口径比较 PSNR。
输出：$FIG_ROOT/figures/（默认 D:/splatanything_figures/figures）
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'figkit'))

import figlib as F  # noqa: E402

F.set_root(os.environ.get('FIG_ROOT', 'D:/splatanything_figures'))

W, H, M = F.W, F.H, F.M


def highlight(d, x, y, w, h, color):
    """关键差异块外环（SPEC §5：外扩 7px、3px、一张图最多 1 处）。"""
    d.rounded_rectangle([x - 7, y - 7, x + w + 7, y + h + 7],
                        radius=14, outline=color, width=3)


def pipeline(d, left, right, c_l, c_r):
    """两栏 6 步，同 y 对齐，返回 (块左x, 块宽, 各步 y)。"""
    bx_l, bx_r = M + F.CARD_INSET, F.W - M - F.CARD_W + F.CARD_INSET
    bw = F.CARD_W - 2 * F.CARD_INSET
    ys = []
    for i in range(F.STEP_N):
        y = F.CARD_Y + F.STEP_TOP + i * (F.STEP_H + F.STEP_GAP)
        ys.append(y)
        F.step(d, bx_l, y, bw, F.STEP_H, c_l, *left[i])
        F.step(d, bx_r, y, bw, F.STEP_H, c_r, *right[i])
        if i < F.STEP_N - 1:
            F.rail(d, bx_l + 12, y + F.STEP_H - 4, y + F.STEP_H + F.STEP_GAP + 4)
            F.rail(d, bx_r + 12, y + F.STEP_H - 4, y + F.STEP_H + F.STEP_GAP + 4)
    return bx_l, bx_r, bw, ys


def fig12():
    im, d = F.canvas()
    F.header(d, 'DA3-GS 与 InfiniSplat v2：同一套摆法，差在「要不要回炉」',
             '两者都是「深度反投影定位置 + 网络只出高斯属性」；分水岭在高斯怎么来、要不要逐场景再优化')

    F.card_head(d, M, F.CARD_Y, F.C_DA3, 'DA3-GS', 'Depth Anything 3 原生高斯头 · 一次前向直出')
    F.card_head(d, W - M - F.CARD_W, F.CARD_Y, F.C_V2, 'InfiniSplat v2',
                'DA3 血统 + 自写回炉精化 · 两段式交付')

    left = [
        ('多视角 RGB（无位姿）', '3~32 图都能进，相机由网络自己估', None, False),
        ('DINOv2 ViT-L/14', '单一预训练大模型提特征，不接 SfM', None, False),
        ('Dual-DPT → 深度 + 射线图', '相机不走外参矩阵，用 depth-ray 表征反解', None, False),
        ('反投影成点云', '深度 × 像素射线 → 世界坐标，位置不参与学习', None, False),
        ('GS-DPT 头：每像素 1 颗', '属性含 SH 2 阶，610 万颗一颗不裁', None, False),
        ('直接导出 PLY', '13.5 s 结束，没有第二步', '1 次前向', False),
    ]
    right = [
        ('多视角 RGB（可选位姿）', 'provided / colmap / da3-pose / pose-free 四模式', None, False),
        ('DA3 backbone（同源）', '内核同一族，钉死 commit 复现', None, False),
        ('逐像素高斯属性', '位置同样来自深度反投影，外观只到 SH DC', None, False),
        ('体素合并', 'voxel = median_scale × 1.4，逐场景自适应', None, False),
        ('opacity 剪枝', 'τ = 0.01，合并后剩 594 万颗', None, False),
        ('回炉精化 3000 步', '位置冻结，只调 SH / scale / σ / quat，163 s', '独有', False),
    ]

    bx_l, bx_r, bw, ys = pipeline(d, left, right, F.C_DA3, F.C_V2)
    highlight(d, bx_r, ys[5], bw, F.STEP_H, F.C_V2)

    F.table(d, M, F.TABLE_Y, W - 2 * M, [
        ('冗余怎么处理', '不融合、不剪枝：像素即高斯\n610 万颗全留（PLY 实存 484 万）',
         '体素加权平均 + opacity 剪枝\n剩 594 万颗，单文件 386 MB'),
        ('外观表达', 'SH 2 阶（全 SH 导出）',
         '仅 SH DC：明暗、高光变化被压平'),
        ('逐场景优化', '无。一次前向即终态',
         '有。固定中心 3000 步，163 s，自洽 +7.30 dB'),
        ('尺度与交付', 'metric 尺度，单 PLY 自带相机',
         'RDNorm 尺度，PLY + cameras.npz + result.json'),
        ('典型失败', '宽基线处摊平成糊；视角 16 人物整体消失',
         '同一空间多份几何副本 → 层间错位重影'),
    ], [('DA3-GS', 796, F.C_DA3), ('InfiniSplat v2', 796, F.C_V2)])

    F.footer(d, '口径：耗时 / 高斯数为 3090 实测（32 视角，佳佳抱枕，FINDINGS §2）；'
                '两列 PSNR 分属不同自洽口径，不可相减。')
    F.save(im, 'fig12_da3gs_vs_v2.png')


def fig13():
    im, d = F.canvas()
    F.header(d, 'InfiniSplat v2 → v3：换基座、加尺度归一、不再回炉',
             'v2 用逐场景回炉换精度；v3 改由 GSN 让一个模型跨场景通用，代价是官方不再提供精化')

    F.card_head(d, M, F.CARD_Y, F.C_V2, 'InfiniSplat v2', 'DA3 内核 · 逐场景尺度 · 自带回炉')
    F.card_head(d, W - M - F.CARD_W, F.CARD_Y, F.C_V3, 'InfiniSplat v3（SplatAnything）',
                'WorldMirror-2 双模型 · GSN 归一 · 仅推理')

    left = [
        ('RGB + 位姿（4 种模式）', 'provided / colmap / da3-pose / pose-free', None, False),
        ('DA3 backbone', '深度 / 相机 / 外观特征一次前馈出齐', None, False),
        ('逐像素高斯属性', '外观仅 SH DC，尺度没有统一口径', None, False),
        ('体素合并（自适应）', 'voxel = median_scale × 1.4，换场景要重算', None, False),
        ('opacity 剪枝', 'τ = 0.01，直接出 SH0 PLY', None, False),
        ('回炉精化 3000 步', '位置冻结只调属性；v3 官方这一步没有了', '已去掉', True),
    ]
    right = [
        ('RGB / pose JSON（2 种）', '位姿模式砍到两种，JSON 按图片 stem 匹配', None, False),
        ('GeometryWorldMirror（教师）', 'CamHead → c2w + K；DPTHead → depth / conf / mask', '教师', False),
        ('GaussianWorldMirror（学生）', '同架构独立权重，不吃教师梯度', '学生', False),
        ('每像素 12 维属性', 'quat4 + scale3 + σ1 + SH 残差 3 + weight1', '换代', False),
        ('GSN 尺度归一', 'spacing = depth / √(fx·fy) → 归一到 1.3228e-3', '新增', False),
        ('固定体素 0.002 合并', '≈1.5 × spacing_target，跨场景不用改参', '新增', False),
    ]

    pipeline(d, left, right, F.C_V2, F.C_V3)

    F.table(d, M, F.TABLE_Y, W - 2 * M, [
        ('几何基座', 'DA3：相机靠 ray 图反解',
         'WorldMirror-2：CamHead 直出 c2w + K\n另给 depth conf 与 mask'),
        ('网络结构', '单模型，一次前馈',
         '双模型串行：几何教师 → 高斯学生\n学生独立权重，不吃教师梯度'),
        ('尺度归一', '无统一口径\nvoxel = median_scale×1.4 逐场景自适应',
         'GSN 把 spacing 归一到 1.3228e-3\n于是 voxel 可固定 0.002'),
        ('位姿输入', '4 种：provided / colmap / da3-pose / pose-free',
         '2 种：rgb 自估 / pose 读 JSON，没有 colmap 模式'),
        ('后处理', '有回炉（自写 02b / 14），自洽 +7.30 dB',
         '官方无精化，注释写明 no training'),
    ], [('InfiniSplat v2', 796, F.C_V2), ('InfiniSplat v3', 796, F.C_V3)])

    F.footer(d, '口径：v2 为 jiajia 32v 全分辨率（FINDINGS §2），v3 为 jiajia_hold 24v 504²'
                '（E-04~E-09）；场景 / 分辨率 / 体素均不同，只比机制不比分数。')
    F.save(im, 'fig13_v2_vs_v3.png')


if __name__ == '__main__':
    fig12()
    fig13()
