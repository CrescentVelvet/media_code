#!/usr/bin/env bash
# run_all.sh — 一键跑通**推荐链路**（阶段二 → 后处理增强）。
#
# 顺序（与 README「阶段 → 脚本」表一致）：
#   02 检测 → 03 匹配 → 04 重建 → 05 对齐 → 06 Avatar 初始化 → 07 Body 初始化
#   → 07a body/scene 切分 → 08 头训练 → 08d scene 定向 finetune
#   → 08e 三分支剪枝 → 09 渲染+后处理增强 → 10 搬回 Windows 盘
#
# 不含（实验性/负结果，见 EXPERIMENTS.md）：08b body finetune、08c body 剪枝、
#   01d 监督图人脸增强。需要时单独跑。
# 前置：01c person mask（无上游产物时）。
#
# 每一步失败即中断；任一步的输入可用同名 env 覆盖（见 README 的 Config 表）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human] 推荐全流程（02 → 10）"
echo "  📁 SOURCE_DIR : $SOURCE_DIR"
echo "  🎭 MASKS      : $PERSON_MASKS_DIR"
echo "  💾 RESULTS_DIR: $RESULTS_DIR"

STEPS=(02_detect_faces.sh 03_match_faces.sh 04_recon_faces.sh
       05_align_3dmm.sh 06_init_avatar_gs.sh 07_init_body_gs.sh
       07a_split_body_scene.sh 08_train.sh 08d_finetune_scene.sh
       08e_prune_all.sh 09_enhance_post.sh)

for s in "${STEPS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "▶  $s"
    echo "════════════════════════════════════════"
    bash "$SCRIPT_DIR/$s"
    if [ $? -ne 0 ]; then
        echo "❌ 中断于 $s"
        exit 1
    fi
done

echo ""
echo "🎉 全流程完成。WSL 上可再跑 10_move_output.sh 搬到 Windows 盘。"
echo "   可选实验分支：08b body finetune（负结果）/ 01d 监督图人脸增强（无增益）"
