# splannequin 加速方案 B：点数预算（10k 配置 + densify 梯度阈值 ×2）
# 与 10k 配置唯一区别 = densify_grad_threshold_fine_init/after 0.0002 → 0.0004。
# 依据：稳态 profile 显示前向+反向占 86% 且近似 ∝ 高斯数（10k_r2 最终 286k）。
# 阈值 ×2 抑制 clone/split，预估点数降到 ~60-75%，净训练 19min → ~13-14min。
# 风险：手部靠点密度，需 04_eval 双口径 + 手部条带 A/B 验收。
# 注：train.py 另有硬编码点数上限 360000（:370/:379），本配置不动它。
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
    iterations = 10_000,
    batch_size = 2,
    coarse_iterations = 1000,
    densify_until_iter = 5_000,
    opacity_reset_interval = 3000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    pruning_interval = 500,
    # 方案 B 核心：fine 阶段 densify 阈值 ×2（coarse 阶段点数少，不动）
    densify_grad_threshold_fine_init = 0.0004,
    densify_grad_threshold_fine_after = 0.0004,

    use_equality=True,

    occlusion_steepness = 3,
    occlusion_from_iter = 3_333,
    l1_occlusion_from_iter = 6_667,
    l1_occlusion_weight = 1e1,
    l2_occlusion_weight = 1e1,

    hidden_steepness = 3,
    hidden_from_iter = 3_333,
    l1_hidden_from_iter = 6_667,
    l1_hidden_weight = 10,
    l2_hidden_weight = 10
)
