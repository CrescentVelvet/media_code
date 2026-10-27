# splannequin 加速方案 A3：等比缩放（30k → 8k，所有相位边界 ×4/15）
# 与 10k 配置（arguments_splannequin_10k.py）唯一区别 = 迭代数 ×0.8。
# 依据：15k→10k 实测边际代价仅 -0.07dB（22.774→22.707 @r1），缩放曲线平缓，
# 再压一档预估净训练 19min → ~15.5min。配套半分辨率（RESOLUTION=2）使用。
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
