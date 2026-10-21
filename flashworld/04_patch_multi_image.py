#!/usr/bin/env python3
"""给官方 FlashWorld 仓打「多图条件」补丁（幂等，带标记，重复跑安全）。

原理（NOTES.md §7c）：条件注入就是 app.py generate() 里
  masks[:, image_index] = 1
  condition_latents[:, :, image_index] = latent
`masks` 变量下游其实没消费——latent 零/非零本身就是掩码。所以多图条件 =
把多张图各自 VAE encode（T=1 单独编码，与官方单图路径逐帧一致），写进多个帧槽位。

打两处：
1. app.py  generate()        ：image/image_index 支持 list（单图路径行为不变）
2. cli.py  process_generation_request()：json 支持
   "image_prompts": [{"path": ..., "index": int}, ...]（与 image_prompt 二选一）

⚠️ 前提：所有条件图同分辨率（同一采集包成立），crop scale 只按第一张图算一次。
⚠️ app.py 的 web 入口 process_generation_request 未打（我们只用 cli.py）。

替换用「起止锚点区间替换」，对块内空白不敏感；起止锚点都要求唯一。

用法（WSL）：python flashworld/04_patch_multi_image.py
还原：cd ~/repos/FlashWorld && git checkout app.py cli.py
"""
import os
import sys

REPO = os.environ.get("FLASHWORLD_DIR") or os.path.expanduser("~/repos/FlashWorld")
MARK = "[multi-image-patch]"

APP_NEW = """            if image is not None:
                # [multi-image-patch] image 可为 Tensor 或 [Tensor,...]；image_index 可为 int 或 [int,...]
                images = list(image) if isinstance(image, (list, tuple)) else [image]
                indices = list(image_index) if isinstance(image_index, (list, tuple)) else [image_index]
                assert len(images) == len(indices), "image 与 image_index 数量不一致"
                for _img, _idx in zip(images, indices):
                    _img = _img.to(self.device)
                    _latent = self.latent_scale_fn(self.vae.encode(
                            _img.unsqueeze(0).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
                        ).latent_dist.sample().to(self.device)).squeeze(2)
                    masks[:, _idx] = 1
                    condition_latents[:, :, _idx] = _latent
                if len(images) > 1:
                    print(f"[multi-image] 条件帧: {indices}")"""

CLI_IMG_NEW = """    # [multi-image-patch] image_prompts: [{"path": ..., "index": int}, ...]，与 image_prompt 二选一
    image_prompts = data.get('image_prompts', None)

    def _crop_resize(pil_img):
        \"\"\"官方 center-crop + resize 逻辑，返回 (tensor, scale, w, h, new_w, new_h)。\"\"\"
        img = pil_img.convert('RGB')
        w, h = img.size
        if image_height / h > image_width / w:
            scale = image_height / h
        else:
            scale = image_width / w
        new_h = int(image_height / scale)
        new_w = int(image_width / scale)
        img = img.crop(((w - new_w) // 2, (h - new_h) // 2,
                        new_w + (w - new_w) // 2, new_h + (h - new_h) // 2)).resize((image_width, image_height))
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 * 2 - 1
        return t, scale, w, h, new_w, new_h

    if image_prompts:
        # 前提：所有条件图同分辨率（同一采集包成立），内参按第一张图的 scale 换算一次
        images, indices = [], []
        scale_applied = False
        for item in image_prompts:
            with Image.open(item['path']) as _im:
                _t, scale, w, h, new_w, new_h = _crop_resize(_im.copy())
            if not scale_applied:
                for camera in cameras:
                    camera['fx'] = camera['fx'] * scale
                    camera['fy'] = camera['fy'] * scale
                    camera['cx'] = (camera['cx'] - (w - new_w) // 2) * scale
                    camera['cy'] = (camera['cy'] - (h - new_h) // 2) * scale
                scale_applied = True
            images.append(_t)
            indices.append(int(item['index']))
        image, image_index = images, indices
        print(f"[multi-image] {len(images)} 张条件图 -> 帧 {indices}")
    elif image_prompt:
        # image_prompt可以是路径和base64
        if os.path.exists(image_prompt):
            image_prompt = Image.open(image_prompt)
        else:
            # image_prompt 可能是 "data:image/png;base64,...."
            if ',' in image_prompt:
                image_prompt = image_prompt.split(',', 1)[1]

            try:
                image_bytes = base64.b64decode(image_prompt)
                image_prompt = Image.open(io.BytesIO(image_bytes))
            except Exception as img_e:
                return {'error': f'Image decode error: {str(img_e)}'}

        image, scale, w, h, new_w, new_h = _crop_resize(image_prompt)

        for camera in cameras:
            camera['fx'] = camera['fx'] * scale
            camera['fy'] = camera['fy'] * scale
            camera['cx'] = (camera['cx'] - (w - new_w) // 2) * scale
            camera['cy'] = (camera['cy'] - (h - new_h) // 2) * scale
    else:
        image = None"""


def replace_span(src, start_anchor, end_anchor, new, label):
    """把 [start_anchor, end_anchor] 区间替换为 new。两个锚点都要求全文件唯一。"""
    if src.count(start_anchor) != 1:
        sys.exit("❌ %s: 起始锚点命中 %d 次，补丁未应用" % (label, src.count(start_anchor)))
    i = src.index(start_anchor)
    if src.count(end_anchor) != 1:
        sys.exit("❌ %s: 结束锚点命中 %d 次，补丁未应用" % (label, src.count(end_anchor)))
    j = src.index(end_anchor, i) + len(end_anchor)
    return src[:i] + new + src[j:]


def patch_file(path, spans, label):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if MARK in src:
        print("  ⏭️  %s 已打过，跳过" % label)
        return
    for start, end, new, sub in spans:
        src = replace_span(src, start, end, new, "%s/%s" % (label, sub))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("  ✅ %s 已打补丁" % label)


def main():
    app_py = os.path.join(REPO, "app.py")
    cli_py = os.path.join(REPO, "cli.py")
    for p in (app_py, cli_py):
        if not os.path.isfile(p):
            sys.exit("❌ 找不到 %s（先跑 00a_setup_env.sh）" % p)
    print("📦 仓库: %s" % REPO)

    # app.py：generate() 的条件注入块（12 空格缩进的是 generate 内的，web 入口是 8 空格，不会误伤）
    patch_file(app_py, [(
        "            if image is not None:",
        "                condition_latents[:, :, image_index] = latent",
        APP_NEW, "generate()")], "app.py 多图条件")

    # cli.py：1) 判空放行 image_prompts；2) 图像预处理块
    patch_file(cli_py, [
        ("    if not image_prompt and text_prompt == \"\":",
         "        return {'error': 'No Prompts provided'}",
         "    if not image_prompt and not data.get('image_prompts') and text_prompt == \"\":  # [multi-image-patch]\n"
         "        return {'error': 'No Prompts provided'}",
         "判空放行"),
        ("    if image_prompt:",
         "        image = None",
         CLI_IMG_NEW, "image_prompts 解析"),
    ], "cli.py")

    import py_compile
    for p in (app_py, cli_py):
        py_compile.compile(p, doraise=True)
    print("✅ 语法检查通过")


if __name__ == "__main__":
    main()
