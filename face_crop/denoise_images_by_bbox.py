#!/usr/bin/env python3
"""
图像主体包围盒裁剪脚本

用途：
通过显著性检测找到图像主体，计算包围盒并裁剪到统一尺寸，用于对齐不同来源的图片进行去噪模型训练。

支持的图片来源：
1. 路径1：jpg格式图片目录
   /data_3d/w00950754/output/renders-videos/测试位姿-新轨迹对比_denoise/Elsa-174-FBX/水豚噜噜3_res720_view90
   - 包含大量的.jpg文件
   - 文件名格式：000.jpg, 001.jpg, 002.jpg 等

2. 路径2：render.jpg格式图片目录
   /data_3d/y00965182/Dataset/dataset_denoise_0609/dataset_dolls_GS/Elsa-174-FBX-horizontal-mp4/Elsa-174-FBX_0a9d593c_1280x1280/result_gs/test_time/denoise_img
   - 包含render.jpg文件
   - 文件名格式：10_theta0.00_phi10.00_zangle180.00_render.jpg 等

裁剪策略：
1. 使用BiRefNet进行显著性分割找到主体
2. 计算主体包围盒（bounding box）
3. 以包围盒中心为中心，根据SCALE_FACTOR调整包围盒大小
4. 裁剪并缩放到固定尺寸（CROP_SIZE）
5. 边界填充使用纯黑色（保持原图片背景）

使用模型：
- BiRefNet-general-20250610_fp16.sbin (默认，512x512检测)
- BiRefNet-general-epoch_244_fp16.sbin (高分辨率，1024x1024检测)

修改配置：
编辑脚本下方的 "===== 参数配置区域 =====" 部分来设置：
- USE_HR: 是否使用高分辨率显著性模型
- CROP_SIZE: 裁剪后图像的边长（正方形），如512、1024等
- SCALE_FACTOR: 包围盒缩放因子，控制主体在输出中的大小（1.0=紧密包围，>1.0=包含更多上下文）
              通过调整此参数可以确保不同来源的图片主体大小一致
- PADDING_MODE: 边界填充模式（推荐'constant'，使用纯黑色填充）
- OUTPUT_NOISE: INPUT_NOISE图片裁剪后输出目录（命名为000_noise, 001_noise等）
- OUTPUT_GT: INPUT_GT图片裁剪后输出目录（命名为000_gt, 001_gt等）
- DEVICE: 设备选择（'cuda' 或 'cpu'）

输出：
- INPUT_NOISE的裁剪图片保存到OUTPUT_NOISE
- INPUT_GT的裁剪图片保存到OUTPUT_GT
- 输出文件名格式：000_noise.png/000_gt.png 等（3位数字编号）
- 所有图片裁剪到统一尺寸CROP_SIZE
- 通过SCALE_FACTOR确保不同来源的图片主体大小一致
- 保持原图片的黑色背景
"""

import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image
from tqdm import tqdm
import logging
import numpy as np
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def get_bounding_boxes(images: list[Tensor], use_HR: bool = False, device=torch.device('cuda')):
    """
    使用BiRefNet模型获取图像主体的包围盒
    
    功能说明：
    1. 加载BiRefNet显著性检测模型（根据use_HR选择512x512或1024x1024版本）
    2. 对每张图像进行显著性分割
    3. 根据分割mask计算主体的最小包围盒
    
    Args:
        images: 图像张量列表，每个张量形状为 (C, H, W)
        use_HR: 是否使用高分辨率模型（1024x1024），默认False使用512x512
        device: 计算设备（cuda/cpu）
    
    Returns:
        bboxes: 包围盒列表，每个包围盒为[xmin, ymin, xmax, ymax]
    """
    torch._C._jit_set_profiling_mode(False)
    torch.jit.optimized_execution(False)
    file_name = "BiRefNet-general-epoch_244_fp16.sbin" if use_HR else "BiRefNet-general-20250610_fp16.sbin"
    size = [1024, 1024] if use_HR else [512, 512]
    load_path = os.path.join("/data_3d/w00950754/model/DollGenRecon/reconstruction_reduce_denoise_accelerate/src/module/mask_generator_weights", file_name)
    model = torch.jit.load(load_path, map_location=device).eval()
    logger.info(f"模型加载完毕！权重路径：{file_name}")
    normalize_opt = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    bboxes = []
    for image in tqdm(images, desc="计算包围盒"):
        image = image.to(device)
        image_resized: Tensor = F.interpolate(image[None], size=size, mode='bilinear', align_corners=True)
        image_normalized: Tensor = normalize_opt(image_resized)
        mask_resized: Tensor = model(image_normalized.half())
        mask: Tensor = F.interpolate(mask_resized, size=image.shape[1:], mode='bilinear', align_corners=True)
        mask = mask.clip(0, 1)[0, 0].contiguous()
        
        # 计算包围盒
        binary_mask = (mask > 0.5).float().cpu().numpy()
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # 如果检测不到主体，使用全图
            bbox = [0, 0, image.shape[2], image.shape[1]]
        else:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox = [xmin, ymin, xmax, ymax]
        
        bboxes.append(bbox)
    
    return bboxes


def crop_image_from_bbox(image: Tensor, bbox: list, crop_size: int, padding_mode='constant', scale_factor=1.0) -> Tensor:
    """
    根据包围盒裁剪图像
    
    Args:
        image: 输入图像张量 (C, H, W)
        bbox: 包围盒 [xmin, ymin, xmax, ymax]
        crop_size: 裁剪尺寸（正方形边长）
        padding_mode: 边界填充模式
        scale_factor: 包围盒缩放因子，用于调整主体大小（1.0=原始包围盒，>1.0=包含更多上下文）
    """
    xmin, ymin, xmax, ymax = bbox
    
    # 计算包围盒的实际大小
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    bbox_max_size = max(bbox_width, bbox_height)
    
    # 使用scale_factor调整包围盒大小，确保不同来源的主体大小一致
    adjusted_bbox_size = bbox_max_size * scale_factor
    
    # 计算包围盒中心
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # 计算裁剪区域，使用调整后的尺寸
    half_size = adjusted_bbox_size / 2
    crop_xmin = int(center_x - half_size)
    crop_ymin = int(center_y - half_size)
    crop_xmax = int(center_x + half_size)
    crop_ymax = int(center_y + half_size)
    
    # 计算需要的填充
    image_h, image_w = image.shape[1], image.shape[2]
    pad_left = max(0, -crop_xmin)
    pad_right = max(0, crop_xmax - image_w)
    pad_top = max(0, -crop_ymin)
    pad_bottom = max(0, crop_ymax - image_h)
    
    # 调整裁剪区域到图像内
    crop_xmin = max(0, crop_xmin)
    crop_ymin = max(0, crop_ymin)
    crop_xmax = min(image_w, crop_xmax)
    crop_ymax = min(image_h, crop_ymax)
    
    # 先裁剪
    cropped = image[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    
    # 再填充到目标尺寸（使用黑色填充）
    padded = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode, value=0)
    
    # 调整到固定输出尺寸
    current_h, current_w = padded.shape[1], padded.shape[2]
    if current_h != crop_size or current_w != crop_size:
        padded = F.interpolate(padded[None], size=(crop_size, crop_size), mode='bilinear', align_corners=True)[0]
    
    return padded


def load_images_from_path_gt(path: str) -> list[tuple[Tensor, str]]:
    """
    从路径加载GT（Ground Truth）图像
    
    功能说明：
    1. 扫描指定路径下所有.jpg文件
    2. 按文件名排序加载图片
    3. 将图片转换为PyTorch张量
    
    Args:
        path: 图片目录路径
    
    Returns:
        images: 图片列表，每个元素为(image_tensor, file_path)元组
    """
    jpg_files = sorted(glob.glob(os.path.join(path, "*.jpg")))
    logger.info(f"从路径1找到 {len(jpg_files)} 张jpg图片")
    
    images = []
    for jpg_file in jpg_files:
        img = Image.open(jpg_file).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        images.append((img_tensor, jpg_file))
    return images


def load_images_from_path_noise(path: str) -> list[tuple[Tensor, str]]:
    """
    从路径加载去噪前图像（render图像）
    
    功能说明：
    1. 扫描指定路径下所有*_render.jpg文件
    2. 从文件名中提取数字ID进行排序（如"10_theta..."按10排序）
    3. 加载图片并转换为PyTorch张量
    
    Args:
        path: 图片目录路径
    
    Returns:
        images: 图片列表，每个元素为(image_tensor, file_path)元组
    """
    render_files = glob.glob(os.path.join(path, "*_render.jpg"))
    
    def extract_id(filepath):
        filename = os.path.basename(filepath)
        try:
            id_str = filename.split('_')[0]
            return int(id_str)
        except (IndexError, ValueError):
            return 0
    
    render_files.sort(key=extract_id)
    logger.info(f"从路径2找到 {len(render_files)} 张render.jpg图片（按ID排序）")
    
    images = []
    for render_file in render_files:
        img = Image.open(render_file).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        images.append((img_tensor, render_file))
    return images


def load_mask_images_from_path_noise(path: str) -> list[tuple[Tensor, str]]:
    """
    从路径加载与render图像对应的mask图像
    
    功能说明：
    1. 扫描指定路径下所有*_mask.jpg文件
    2. 从文件名中提取数字ID进行排序（与render图像排序一致）
    3. 加载图片并转换为PyTorch张量
    
    Args:
        path: 图片目录路径
    
    Returns:
        images: 图片列表，每个元素为(image_tensor, file_path)元组
    """
    mask_files = glob.glob(os.path.join(path, "*_mask.jpg"))
    
    def extract_id(filepath):
        filename = os.path.basename(filepath)
        try:
            id_str = filename.split('_')[0]
            return int(id_str)
        except (IndexError, ValueError):
            return 0
    
    mask_files.sort(key=extract_id)
    logger.info(f"从路径找到 {len(mask_files)} 张mask.jpg图片（按ID排序）")
    
    images = []
    for mask_file in mask_files:
        img = Image.open(mask_file).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        images.append((img_tensor, mask_file))
    return images


def save_cropped_images(images: list[Tensor], image_files: list[str], output_dir: str,
                        name_format: str | None = None, start_idx: int = 0, folder_prefix: str = ''):
    """
    保存裁剪后的图片

    Args:
        images: 裁剪后的图片张量列表
        image_files: 原始图片文件路径列表
        output_dir: 输出目录
        name_format: 命名格式 (可选，如 'noise' 或 'gt'，为None时按编号命名)
        start_idx: 起始索引（用于多条路径连续编号）
        folder_prefix: 文件夹名称前缀（用于区分不同来源的图像）
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (cropped_img, img_file) in enumerate(zip(images, image_files), start=start_idx):
        img_pil = transforms.ToPILImage()(cropped_img)
        # 格式化为3位数字，如000, 001, 002等
        if folder_prefix:
            filename = f"{folder_prefix}_{idx:03d}.png"
        else:
            filename = f"{idx:03d}.png" if name_format is None else f"{idx:03d}_{name_format}.png"
        output_path = os.path.join(output_dir, filename)
        img_pil.save(output_path)

    format_desc = "编号格式" if name_format is None else f"{name_format}格式"
    logger.info(f"已保存 {len(images)} 张{format_desc}图片到 {output_dir}")


def process_images(images: list[tuple[Tensor, str]], output_dir: str, name_format: str | None,
                    use_HR: bool, crop_size: int, padding_mode: str, device: torch.device,
                    start_idx: int = 0, scale_factor: float = 1.0, folder_prefix: str = ''):
    """
    处理一组图片并保存裁剪结果

    Args:
        images: 图片张量和文件路径列表
        output_dir: 输出目录
        name_format: 命名格式 (可选，如 'noise' 或 'gt'，为None时按编号命名)
        use_HR: 是否使用高分辨率模型
        crop_size: 裁剪尺寸
        padding_mode: 填充模式
        device: 设备
        start_idx: 起始索引
        scale_factor: 包围盒缩放因子
        folder_prefix: 文件夹名称前缀（用于区分不同来源的图像）
    """
    if not images:
        logger.error("未找到任何图片!")
        return

    image_tensors = [img for img, _ in images]
    image_files = [img_file for _, img_file in images]

    format_desc = "编号格式" if name_format is None else f"{name_format}格式"
    logger.info(f"开始处理 {len(images)} 张图片，命名为{format_desc}")

    # 获取包围盒
    bboxes = get_bounding_boxes(image_tensors, use_HR=use_HR, device=device)

    # 裁剪图像
    cropped_images = []
    for img, bbox in tqdm(zip(image_tensors, bboxes), desc="裁剪图像", total=len(image_tensors)):
        cropped = crop_image_from_bbox(img, bbox, crop_size, padding_mode=padding_mode, scale_factor=scale_factor)
        cropped_images.append(cropped)

    # 保存裁剪结果
    save_cropped_images(cropped_images, image_files, output_dir, name_format, start_idx, folder_prefix)

    return bboxes


def process_mask_with_bbox(mask_images: list[tuple[Tensor, str]], bboxes: list[list],
                           output_dir: str, crop_size: int, padding_mode: str,
                           start_idx: int = 0, folder_prefix: str = '', scale_factor: float = 1.0):
    """
    使用已有的包围盒裁剪mask图像
    
    mask图像使用与render图像相同的包围盒进行裁剪，确保对齐。
    
    Args:
        mask_images: mask图片张量和文件路径列表
        bboxes: 从render图像计算得到的包围盒列表
        output_dir: 输出目录
        crop_size: 裁剪尺寸
        padding_mode: 填充模式
        start_idx: 起始索引
        folder_prefix: 文件夹名称前缀
        scale_factor: 包围盒缩放因子（应与render图像一致）
    """
    if not mask_images:
        logger.warning("未找到任何mask图片!")
        return

    if len(mask_images) != len(bboxes):
        logger.warning(f"mask图片数量({len(mask_images)})与包围盒数量({len(bboxes)})不匹配，跳过mask处理")
        return

    mask_tensors = [img for img, _ in mask_images]
    mask_files = [f for _, f in mask_images]

    cropped_masks = []
    for img, bbox in tqdm(zip(mask_tensors, bboxes), desc="裁剪mask图像", total=len(mask_tensors)):
        cropped = crop_image_from_bbox(img, bbox, crop_size, padding_mode=padding_mode, scale_factor=scale_factor)
        cropped_masks.append(cropped)

    save_cropped_images(cropped_masks, mask_files, output_dir, None, start_idx, folder_prefix)


def test_one_image():
    """
    单文件夹模式处理函数
    
    功能说明：
    1. 处理单个文件夹内的图像对（noise和gt）
    2. 分别加载、裁剪并保存noise和gt图像
    3. 输出到两个独立目录，图像名为{idx:03d}_{noise/gt}.png格式
    
    使用场景：
    - 需要快速测试处理效果时使用
    - 处理单个数据子集时使用
    
    配置说明：
    修改下方"参数配置区域"的变量来调整处理参数
    """
    
    # ===== 参数配置区域 ===== 
    USE_HR = True
    CROP_SIZE = 720
    SCALE_FACTOR = 1.2  # 包围盒缩放因子，控制主体在输出中的大小（1.0=紧密包围，>1.0=包含更多上下文）
    PADDING_MODE = 'constant'
    DEVICE = 'cuda'

    INPUT_NOISE = '/data_3d/y00965182/Dataset/dataset_denoise_0609/dataset_dolls_GS/old_test/Elsa-174-FBX-horizontal-mp4/Elsa-174-FBX_0a9d593c_1280x1280/result_gs/test_time/denoise_img'
    OUTPUT_NOISE = "/data_3d/y00965182/Dataset/dataset_denoise_0609/dataset_dolls_GS/old_test/Elsa-174-FBX-horizontal-mp4-cropped-render"

    INPUT_GT = '/data_3d/w00950754/output/renders-videos/测试位姿-新轨迹对比_denoise/Elsa-174-FBX/水豚噜噜3_res720_view90'
    OUTPUT_GT = "/data_3d/y00965182/Dataset/dataset_denoise_0609/dataset_dolls_GS/old_test/Elsa-174-FBX-horizontal-mp4-cropped-gt"
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"裁剪尺寸: {CROP_SIZE}x{CROP_SIZE}")
    
    # 处理INPUT_NOISE，命名为noise格式
    logger.info("=" * 50)
    logger.info("🚀 开始处理INPUT_NOISE的图片（命名为noise格式）...")
    logger.info("=" * 50)
    images_path1 = load_images_from_path_noise(INPUT_NOISE)
    process_images(images_path1, OUTPUT_NOISE, 'noise', USE_HR, CROP_SIZE, PADDING_MODE, device, start_idx=0, scale_factor=SCALE_FACTOR)
    
    # 处理INPUT_GT，命名为gt格式
    logger.info("=" * 50)
    logger.info("🚀 开始处理INPUT_GT的图片（命名为gt格式）...")
    logger.info("=" * 50)
    images_path2 = load_images_from_path_gt(INPUT_GT)
    process_images(images_path2, OUTPUT_GT, 'gt', USE_HR, CROP_SIZE, PADDING_MODE, device, start_idx=0, scale_factor=SCALE_FACTOR)
    
    logger.info("=" * 50)
    logger.info("✅ 所有图片处理完成！")
    logger.info(f"📂 OUTPUT_NOISE(noise)结果保存在: {OUTPUT_NOISE}")
    logger.info(f"📂 OUTPUT_GT(gt)结果保存在: {OUTPUT_GT}")


def batch_process():
    """
    批量模式处理函数
    
    功能说明：
    1. 批量处理多个文件夹中的图像对
    2. 自动匹配INPUT_BASE_NOISE和INPUT_BASE_GT下的所有子文件夹
    3. 通过提取文件夹描述名称（去掉分辨率和视图信息）进行匹配
    4. NOISE图像路径：{INPUT_BASE_NOISE}/{folder_name}/result_gs/test_time/denoise_img 或 {INPUT_BASE_NOISE}/{folder_name}
       GT图像路径：{INPUT_BASE_GT}/{folder_name}
    5. 只处理两边都存在且路径有效的文件夹
    6. 每个文件夹的图像使用文件夹描述名称作为前缀区分
    7. 输出到统一的render和gt目录，便于组织数据集
    
    路径匹配规则：
    例如：
    - NOISE: ".../dataset_pets_GS/black_cat_res1280_view81/result_gs/test_time/denoise_img"
    - GT:    ".../dataset_pets_GS/black_cat_res720_view90"
    - 描述名称: "black_cat"（去掉_resX_viewY部分）
    - 数据集识别: "pets"（从路径自动提取）
    - 最终图像名（启用前缀）: "pets_black_cat_000.png"
    - 最终图像名（禁用前缀）: "black_cat_000.png"
    
    输出结构（启用数据集前缀）：
    OUTPUT_BASE/
    ├── render/          # 所有去噪前图像
    │   ├── pets_black_cat_000.png
    │   ├── pets_black_cat_001.png
    │   ├── dolls_艾莎1_000.png
    │   └── ...
    ├── gt/              # 所有GT图像
    │   ├── pets_black_cat_000.png
    │   ├── pets_black_cat_001.png
    │   ├── dolls_艾莎1_000.png
    │   └── ...
    └── mask/            # 所有mask图像（使用与render相同的包围盒裁剪）
        ├── pets_black_cat_000.png
        ├── pets_black_cat_001.png
        ├── dolls_艾莎1_000.png
        └── ...
    
    特点：
    - noise和gt对应图像同名（通过相同start_idx和folder_prefix实现）
    - 图像名格式：{dataset_name}_{folder_prefix}_{idx:03d}.png 或 {folder_prefix}_{idx:03d}.png
    - 支持数据集前缀配置，避免不同数据集中同名文件夹冲突
    - 自动从base路径提取数据集名称
    - folder_prefix取文件夹名去掉_resX_viewY后的描述名称
    - 自动处理NOISE图像可能在子目录的情况
    
    配置说明：
    - USE_DATASET_PREFIX: 是否启用数据集前缀（默认True）
    - 修改下方"参数配置区域"的变量来调整处理参数和路径
    
    使用场景：
    - 需要一次性处理大量数据时使用
    - 构建完整去噪训练数据集时使用
    - 处理多个不同数据集，需要避免同名文件夹冲突时使用（启用数据集前缀）
    - 处理单个数据集，希望输出文件名简洁时使用（禁用数据集前缀）
    
    配置说明：
    修改下方"参数配置区域"的变量来调整处理参数和路径：
    - USE_HR: 是否使用高分辨率显著性模型
    - CROP_SIZE: 裁剪后图像的边长
    - SCALE_FACTOR: 包围盒缩放因子
    - PADDING_MODE: 边界填充模式
    - DEVICE: 设备选择
    - USE_DATASET_PREFIX: 是否启用数据集前缀避免同名冲突
    - INPUT_BASE_NOISE/GT: 输入基础路径
    - OUTPUT_BASE: 输出基础路径
    
    启动方式：
    python script.py --batch
    """
    # ===== 参数配置区域 =====
    USE_HR = True
    CROP_SIZE = 720
    SCALE_FACTOR = 1.2
    PADDING_MODE = 'constant'
    DEVICE = 'cuda'
    
    # 是否使用数据集前缀避免不同数据集中的同名文件夹冲突
    # True: 图像名格式为 "dataset_foldername_000.png"
    # False: 图像名格式为 "foldername_000.png"（可能在不同数据集中产生冲突）
    USE_DATASET_PREFIX = True

    DATASETS = [
        # 玩偶去噪数据集
        {'names': ['Elsa-174-FBX', 'Baiyu-104-FBX', 'Mixamo-102-FBX'],
         'input_base_noise': '/data_3d/y00965182/Dataset/dataset_denoise_0702/dataset_dolls_GS',
         'input_base_gt': '/data_3d/w00950754/output/renders-videos/测试位姿-新轨迹对比_denoise',
         'output_base': '/data_3d/y00965182/Dataset/dataset_denoise_0702/dataset_dolls_GS-dataset'},
        # 萌宠去噪数据集
        {'names': ['Cats-Sketch-56-GLB', 'Cats-UE-21-GLB', 'CatsDogs-64-FBX',
                   'Dogs-Blenderkit-25-BLD', 'Dogs-Sketch-102-GLB', 'Dogs-UE-58-GLB'],
         'input_base_noise': '/data_3d/y00965182/Dataset/dataset_denoise_0702/dataset_pets_GS',
         'input_base_gt': '/data_3d/w00950754/output/renders-videos/测试位姿-新轨迹对比_denoise',
         'output_base': '/data_3d/y00965182/Dataset/dataset_denoise_0702/dataset_pets_GS-dataset'},
    ]
    # 输出基础路径，会自动创建render和gt子文件夹

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"裁剪尺寸: {CROP_SIZE}x{CROP_SIZE}")

    for ds_idx, ds_config in enumerate(DATASETS):
        logger.info("=" * 70)
        logger.info(f"🚀 开始处理数据集 [{ds_idx+1}/{len(DATASETS)}]")
        logger.info("=" * 70)

        for NAME in ds_config['names']:
            INPUT_BASE_NOISE = os.path.join(ds_config['input_base_noise'], NAME)
            INPUT_BASE_GT = os.path.join(ds_config['input_base_gt'], NAME)
            OUTPUT_BASE = ds_config['output_base']

            logger.info("=" * 60)
            logger.info(f"📦 处理: {NAME}")
            logger.info(f"   NOISE: {INPUT_BASE_NOISE}")
            logger.info(f"   GT:    {INPUT_BASE_GT}")
            logger.info(f"   OUT:   {OUTPUT_BASE}")
            logger.info("=" * 60)

            if not os.path.exists(INPUT_BASE_NOISE):
                logger.warning(f"NOISE路径不存在: {INPUT_BASE_NOISE}，跳过")
                continue
            if not os.path.exists(INPUT_BASE_GT):
                logger.warning(f"GT路径不存在: {INPUT_BASE_GT}，跳过")
                continue

            _process_single_dataset(
                INPUT_BASE_NOISE, INPUT_BASE_GT, OUTPUT_BASE,
                USE_HR, CROP_SIZE, SCALE_FACTOR, PADDING_MODE,
                USE_DATASET_PREFIX, device
            )

    logger.info("=" * 70)
    logger.info("✅ 所有数据集处理完成！")


def _process_single_dataset(INPUT_BASE_NOISE, INPUT_BASE_GT, OUTPUT_BASE,
                             USE_HR, CROP_SIZE, SCALE_FACTOR, PADDING_MODE,
                             USE_DATASET_PREFIX, device):
    output_render_dir = os.path.join(OUTPUT_BASE, 'render')
    output_gt_dir = os.path.join(OUTPUT_BASE, 'gt')
    output_mask_dir = os.path.join(OUTPUT_BASE, 'mask')

    logger.info("=" * 50)
    logger.info("📂 开始批量处理图像文件夹...")
    logger.info("=" * 50)

    # 获取INPUT_BASE_NOISE下的所有子文件夹
    noise_folders = [f for f in os.listdir(INPUT_BASE_NOISE) if os.path.isdir(os.path.join(INPUT_BASE_NOISE, f))]
    logger.info(f"INPUT_BASE_NOISE找到 {len(noise_folders)} 个子文件夹")

    # 获取INPUT_BASE_GT下的所有子文件夹
    gt_folders = [f for f in os.listdir(INPUT_BASE_GT) if os.path.isdir(os.path.join(INPUT_BASE_GT, f))]
    logger.info(f"INPUT_BASE_GT找到 {len(gt_folders)} 个子文件夹")

    # 提取文件夹的描述名称（去掉分辨率和视图信息）
    def extract_folder_name(folder_name):
        """
        从文件夹名中提取描述名称
        逻辑：只移除最后的 _resXXXX_viewXX 后缀，保留其他所有内容
        
        例如：
        - 'black_cat_res1280_view81' -> 'black_cat'
        - 'black_and_white_cat_walking_res1280_view81' -> 'black_and_white_cat_walking'
        - 'cat_3d_res1280_view81' -> 'cat_3d'
        - 'sphynx_cat (2)_res1280_view81' -> 'sphynx_cat (2)'
        """
        # 使用正则表达式匹配最后的 _res\d+_view\d+ 模式并移除
        pattern = r'_res\d+_view\d+$'
        match = re.search(pattern, folder_name)
        if match:
            return folder_name[:match.start()]
        return folder_name

    # 从base路径提取数据集名称
    def extract_dataset_name(base_path):
        """
        从base路径提取数据集名称作为前缀
        优先使用数据集标识符，否则使用路径最后一部分
        
        例如：
        - '/.../dataset_denoise_0609/dataset_dolls_GS/Elsa-174-FBX' -> 'dolls'
        - '/.../dataset_pets_GS/Cats-Sketch-56-GLB' -> 'pets'
        - '/.../other_folder' -> 'other'
        """
        path_parts = base_path.split('/')
        for part in reversed(path_parts):
            # 查找包含dataset标识符的文件夹名
            if 'dataset_' in part and '_GS' in part:
                # 提取GS前的部分作为数据集名
                return part.split('_GS')[0].replace('dataset_', '')
            # 如果没有dataset标识符，使用最后一个非空部分
            if part and not part.startswith('/') and part != '.':
                return part
        return 'dataset'  # 兜底返回

    # 构建noise文件夹的名称映射
    noise_name_map = {extract_folder_name(f): f for f in noise_folders}
    gt_name_map = {extract_folder_name(f): f for f in gt_folders}

    # 提取数据集名称（用于区分不同数据集中的同名文件夹）
    dataset_name = extract_dataset_name(INPUT_BASE_NOISE) if USE_DATASET_PREFIX else ""
    if dataset_name:
        logger.info(f"📊 数据集识别: {dataset_name}")
    logger.info(f"   NOISE路径: {INPUT_BASE_NOISE}")
    logger.info(f"   GT路径: {INPUT_BASE_GT}")

    # 检查是否有重复的描述名称
    noise_name_counts = {}
    gt_name_counts = {}

    for folder in noise_folders:
        name = extract_folder_name(folder)
        noise_name_counts[name] = noise_name_counts.get(name, 0) + 1

    for folder in gt_folders:
        name = extract_folder_name(folder)
        gt_name_counts[name] = gt_name_counts.get(name, 0) + 1

    # 找出重复的描述名称
    noise_duplicates = {name: count for name, count in noise_name_counts.items() if count > 1}
    gt_duplicates = {name: count for name, count in gt_name_counts.items() if count > 1}

    if noise_duplicates:
        logger.warning(f"⚠️  NOISE文件夹中存在重复的描述名称:")
        for name, count in sorted(noise_duplicates.items()):
            logger.warning(f"  - '{name}' 重复 {count} 次")
            # 显示所有重复的文件夹
            dup_folders = [f for f in noise_folders if extract_folder_name(f) == name]
            for folder in dup_folders:
                logger.warning(f"      -> {folder}")

    if gt_duplicates:
        logger.warning(f"⚠️  GT文件夹中存在重复的描述名称:")
        for name, count in sorted(gt_duplicates.items()):
            logger.warning(f"  - '{name}' 重复 {count} 次")
            # 显示所有重复的文件夹
            dup_folders = [f for f in gt_folders if extract_folder_name(f) == name]
            for folder in dup_folders:
                logger.warning(f"      -> {folder}")

    # 找到两边都存在的描述名称
    common_names = sorted(set(noise_name_map.keys()) & set(gt_name_map.keys()))
    logger.info(f"两边都存在的文件夹数量: {len(common_names)}")

    if len(common_names) == 0:
        logger.error("没有找到两边都存在的文件夹！")
        logger.info("NOISE文件夹样例:")
        for f in list(noise_name_map.keys())[:5]:
            logger.info(f"  - {f} -> {noise_name_map[f]}")
        logger.info("GT文件夹样例:")
        for f in list(gt_name_map.keys())[:5]:
            logger.info(f"  - {f} -> {gt_name_map[f]}")
        return

    # 输出无法匹配的文件夹名称，便于查错
    noise_only = sorted(set(noise_name_map.keys()) - set(gt_name_map.keys()))
    gt_only = sorted(set(gt_name_map.keys()) - set(noise_name_map.keys()))

    logger.info("=" * 50)
    logger.info("🔍 匹配情况统计")
    logger.info(f"  - NOISE文件夹总数: {len(noise_folders)}个")
    logger.info(f"  - NOISE唯一描述名: {len(set(noise_name_map.keys()))}个")
    logger.info(f"  - GT文件夹总数: {len(gt_folders)}个")
    logger.info(f"  - GT唯一描述名: {len(set(gt_name_map.keys()))}个")
    logger.info(f"  - 成功匹配: {len(common_names)}个")
    logger.info(f"  - NOISE独有: {len(noise_only)}个")
    logger.info(f"  - GT独有: {len(gt_only)}个")
    logger.info("=" * 50)

    if noise_only:
        logger.warning(f"⚠️  在NOISE中存在但在GT中不存在的文件夹（{len(noise_only)}个）:")
        for name in noise_only:
            logger.warning(f"  - {name} -> {noise_name_map[name]}")

    if gt_only:
        logger.warning(f"⚠️  在GT中存在但在NOISE中不存在的文件夹（{len(gt_only)}个）:")
        for name in gt_only:
            logger.warning(f"  - {name} -> {gt_name_map[name]}")

    # 计数器
    total_processed = 0

    for name in tqdm(common_names, desc="处理文件夹", colour="GREEN"):
        logger.info("=" * 50)
        logger.info(f"📁 处理文件夹: {name}")
        logger.info("=" * 50)

        noise_folder = noise_name_map[name]
        gt_folder = gt_name_map[name]

        # NOISE图像在子文件夹内的 result_gs/test_time/denoise_img 目录
        input_noise_path = os.path.join(INPUT_BASE_NOISE, noise_folder, 'result_gs', 'test_time', 'denoise_img')
        # GT图像直接在子文件夹下
        input_gt_path = os.path.join(INPUT_BASE_GT, gt_folder)

        # 如果NOISE路径不存在，尝试直接在文件夹下查找
        if not os.path.exists(input_noise_path):
            input_noise_path = os.path.join(INPUT_BASE_NOISE, noise_folder)
            logger.info(f"NOISE路径不存在，尝试使用: {input_noise_path}")

        # 检查路径是否存在
        if not os.path.exists(input_noise_path):
            logger.warning(f"NOISE路径不存在: {input_noise_path}，跳过此文件夹")
            continue
        if not os.path.exists(input_gt_path):
            logger.warning(f"GT路径不存在: {input_gt_path}，跳过此文件夹")
            continue

        # 使用文件夹描述名称作为图像名前缀，根据配置决定是否添加数据集前缀
        if USE_DATASET_PREFIX and dataset_name:
            folder_prefix = f"{dataset_name}_{name}"
        else:
            folder_prefix = name

        # 处理当前文件夹的起始索引
        current_start_idx = total_processed

        # 处理noise图像
        logger.info(f"处理noise图像: {input_noise_path}")
        images_noise = load_images_from_path_noise(input_noise_path)
        bboxes_noise = None
        if images_noise:
            bboxes_noise = process_images(images_noise, output_render_dir, None, USE_HR, CROP_SIZE,
                          PADDING_MODE, device, start_idx=current_start_idx,
                          scale_factor=SCALE_FACTOR, folder_prefix=folder_prefix)

        # 处理mask图像（使用与render图像相同的包围盒）
        logger.info(f"处理mask图像: {input_noise_path}")
        images_mask = load_mask_images_from_path_noise(input_noise_path)
        if images_mask and bboxes_noise:
            process_mask_with_bbox(images_mask, bboxes_noise, output_mask_dir, CROP_SIZE,
                                   PADDING_MODE, start_idx=current_start_idx,
                                   folder_prefix=folder_prefix, scale_factor=SCALE_FACTOR)

        # 处理gt图像（和noise图像使用相同的起始索引，这样对应图像同名）
        logger.info(f"处理gt图像: {input_gt_path}")
        images_gt = load_images_from_path_gt(input_gt_path)
        if images_gt:
            process_images(images_gt, output_gt_dir, None, USE_HR, CROP_SIZE,
                          PADDING_MODE, device, start_idx=current_start_idx,
                          scale_factor=SCALE_FACTOR, folder_prefix=folder_prefix)

        # 更新总处理数量（取两者中较大的）
        total_processed += max(len(images_noise), len(images_gt))

    logger.info("=" * 50)
    logger.info("✅ 所有文件夹处理完成！")
    logger.info(f"📂 render输出路径: {output_render_dir}")
    logger.info(f"📂 gt输出路径: {output_gt_dir}")
    logger.info(f"📂 mask输出路径: {output_mask_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_process()
    else:
        test_one_image()