#!/usr/bin/env bash
# AUDIT-2 运行器：渲染分解（all / self / self÷α / other）+ 与 94 交付图的闸门
#
# 用法：bash compare_4methods/audit_2_render_decomp.sh [direct|res1|res2]
# gsplat 是 JIT 编译的，必须 CUDA_HOME/CPATH/LIBRARY_PATH 三样到位（同 94_render.sh），
# 且不要依赖 $CONDA_PREFIX（经 wsl.exe 传参会被吃掉）。
set -eo pipefail
ENVS="${QUERYSPLAT_ENV:-$HOME/miniconda3/envs/querysplat}"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENVS"
else
  export PATH="$ENVS/bin:$PATH"
fi
export CUDA_HOME="$ENVS"
export CPATH="$ENVS/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENVS/targets/x86_64-linux/lib"

B="$HOME/output/da3_gs_results"
NPZ="$B/佳佳抱枕_p1m/cameras.npz"
INIT_PLY="$B/佳佳抱枕_p1m/gs_ply/0000.ply"
case "${1:-res1}" in
  direct) PLY="$B/佳佳抱枕_p1m/gs_ply/0000.ply";              OUT="$B/佳佳抱枕_p1m/audit_direct"; REF94="" ;;
  res1)   PLY="$B/佳佳抱枕_res_ab/final_res1/point_cloud/iteration_2000/point_cloud.ply"
          OUT="$B/佳佳抱枕_res_ab/audit_res1";                REF94="$B/佳佳抱枕_res_ab/final_res1/render_378x504" ;;
  res2)   PLY="$B/佳佳抱枕_res_ab/final_res2/point_cloud/iteration_2000/point_cloud.ply"
          OUT="$B/佳佳抱枕_res_ab/audit_res2";                REF94="$B/佳佳抱枕_res_ab/final_res2/render_378x504" ;;
  *) echo "未知标签 $1（direct|res1|res2）"; exit 2 ;;
esac
echo "PLY=$PLY"
echo "OUT=$OUT"
PLY="$PLY" NPZ="$NPZ" OUT="$OUT" REF94="$REF94" INIT_PLY="$INIT_PLY" \
  VIEWS="${VIEWS:-0,8,16}" \
  exec python "$(dirname "$0")/audit_2_render_decomp.py"
