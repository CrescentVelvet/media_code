# splannequin 加速方案 A+B 组合：8k 等比 + densify 阈值 ×2
# 依据：A（8k）实测零质量损失（22.549 @r1），B（dgx2）点数 -40% 仅 -0.10dB（22.453 @r1）。
# 组合预估：净训练 ~10min，质量 ~22.45dB @r1（两方案代价近似可加）。
# ⚠️ 本次为「外部显存占用」鉴别实验：全程挂双侧 GPU/内存监控
#    （WSL 侧 07_monitor_gpu.sh + Windows 侧 nvidia-smi 循环）。
# 用法：RESOLUTION=2 CONFIG_FILE=<本文件路径> bash 02_train.sh
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False

)
OptimizationParams = dict(
    dataloader=True,
    iterations = 8_000,
    batch_size = 2,
    coarse_iterations = 800,
    densify_until_iter = 4_000,
    # 保持官方值（不随总迭代缩放）：densify_until=4000 时本配置仅 3000 一次 reset。
    opacity_reset_interval = 3000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    pruning_interval = 500,
    # 方案 B：fine 阶段 densify 阈值 ×2
    densify_grad_threshold_fine_init = 0.0004,
    densify_grad_threshold_fine_after = 0.0004,

    use_equality=True,

    occlusion_steepness = 3,
    occlusion_from_iter = 2_667,
    l1_occlusion_from_iter = 5_333,
    l1_occlusion_weight = 1e1,
    l2_occlusion_weight = 1e1,

    hidden_steepness = 3,
    hidden_from_iter = 2_667,
    l1_hidden_from_iter = 5_333,
    l1_hidden_weight = 10,
    l2_hidden_weight = 10
)
