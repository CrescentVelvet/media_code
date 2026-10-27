# splannequin 加速方案 A2：等比缩放（30k → 10k，所有相位边界 ×1/3）
# 与官方 arguments/splannequin.py 唯一区别 = OptimizationParams 中所有
# 迭代数相关常量 ×1/3。方法相位结构（coarse→稠密化→hidden/occlusion 检测
# →L1/L2 锚定）完整保留。配套半分辨率（RESOLUTION=2）使用。
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
    # 保持官方值（不随总迭代缩放）：实测崩溃集中在 opacity reset 后的恢复期，
    # 减少 reset 次数 = 减少危险窗口。densify_until=5000 时本配置仅 3000 一次 reset。
    opacity_reset_interval = 3000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    pruning_interval = 500,

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
