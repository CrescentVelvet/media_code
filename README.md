# media_code
Code used for work in the blue zone.
- ♠️♥️♣️♦️
- 🌈🔥💧☄️❄️⚡🫧💦⛈️
- 👑🏆🥇🥈🥉🎯🔮♨️🎡🚨⚜️
- ✨🌟🎃👓🚀🧨🎈⛩️🧣🎗️🎀👠
- 💎📀🪙💵💡🍉🔍💥🧪
- ❓❗⁉️⭕❌✅♻️
- ⛔⚠️☢️😍🥰😭

## Algorithms
- [`triposplat/`](triposplat/) — one-click TripoSplat image→3D-Gaussian inference. See its [README](triposplat/README.md).
- [`trellis-2/`](trellis-2/) — one-click TRELLIS.2 image→3D (4B) inference: GLB + PBR turntable video. See its [README](trellis-2/README.md).
- [`vggt-omega/`](vggt-omega/) — one-click VGGT-Omega feed-forward scene reconstruction (cameras + depth → point cloud). See its [README](vggt-omega/README.md).
- [`hunyuanvideo-1.5/`](hunyuanvideo-1.5/) — one-click HunyuanVideo-1.5 video generation: T2V/I2V inference **and** fine-tuning on your own dataset. See its [README](hunyuanvideo-1.5/README.md).
- [`hypir/`](hypir/) — one-click HYPIR image restoration/super-resolution: inference, dataset construction, **and** LoRA training on your own images. See its [README](hypir/README.md).
- [`qwen3vl/`](qwen3vl/) — one-click Qwen3-VL image-to-text (图生文) batch inference: VLM captioning / VQA over a folder of images → per-image `.txt`. See its [README](qwen3vl/README.md).
- [`flux1/`](flux1/) — one-click FLUX.1 text-to-image (文生图) batch inference: prompts (single or per-line file) → PNGs via diffusers `FluxPipeline` (schnell default; dev gated). See its [README](flux1/README.md).
- [`face_crop/`](face_crop/) — one-click batch face detection & cropping (MediaPipe BlazeFace): multi-face per image → `<stem>_faceN.jpg` sorted left-to-right, resumable CSV log. See its [README](face_crop/README.md).
- [`retouchformer/`](retouchformer/) — one-click RetouchFormer (AAAI 2024) face retouching batch inference: folder of face images → retouched 512×512 PNGs (GPEN + VRT selective self-attention, single forward; Baidu-only weights). See its [README](retouchformer/README.md).
- [`wan22/`](wan22/) — one-click Wan2.2-TI2V-5B (DiffSynth-Studio) text/image-to-video generation: T2V/I2V inference **and** LoRA fine-tuning on your own video dataset. See its [README](wan22/README.md).
- [`wan22_rotate/`](wan22_rotate/) — orbit images → 360° rotation video: SAM 3D Body picks the front-facing image & segments the person (white bg) → Wan2.2 I2V + LoRA generates a 360° rotation video. See its [README](wan22_rotate/README.md).
- [`pi3_3dgs/`](pi3_3dgs/) — video → 3D reconstruction: Pi3 (π³, ICLR 2026) feed-forward pose+point estimation → COLMAP export → 2D Gaussian Splatting (SIGGRAPH 2024) training + mesh. See its [README](pi3_3dgs/README.md).
- [`minimax_h3/`](minimax_h3/) — one-click MiniMax-H3 (H3-Base 768p) omni-modal video+audio reproduction via SGLang Diffusion: serve T2VA/FL2VA/Ref2VA + reproduce the three official 768p cases. See its [README](minimax_h3/README.md).
