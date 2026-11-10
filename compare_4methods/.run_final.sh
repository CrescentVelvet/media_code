#!/bin/bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/querysplat"
export CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/targets/x86_64-linux/include" LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"
export G3R_PLY=/home/velvet/output/gen3r_results/02_infer_jiajia/1view/2026-11-10-17-42-00/3dgs_7000/point_cloud/iteration_7000/point_cloud.ply
export G3R_COLMAP=/home/velvet/output/gen3r_results/02_infer_jiajia/1view/2026-11-10-17-42-00/colmap_scene
export SS_PLY=$HOME/output/scene_splatter_results/02_infer/2026-11-10-20-29-37/scene.ply
export SS_TRAJ=$HOME/output/scene_splatter_results/traj/jiajia_orbit_220_264.pkl.gz
export OUT=$HOME/output/compare_4methods_video/three_way_final NFRAMES=144 FPS=24
cd /tmp && /home/velvet/miniconda3/envs/querysplat/bin/python /mnt/c/code/media_code/compare_4methods/audit_33_three_way_video.py 2>&1 | grep -v TORCH_CUDA
