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
- [`vggt_human/`](vggt_human/) — VGGT-Omega feed-forward recon (poses+depth→point cloud) → COLMAP → original 3DGS training: combines fast feed-forward init with optimization-based quality. See its [README](vggt_human/README.md).
- [`hunyuanvideo-1.5/`](hunyuanvideo-1.5/) — one-click HunyuanVideo-1.5 video generation: T2V/I2V inference **and** fine-tuning on your own dataset. See its [README](hunyuanvideo-1.5/README.md).
- [`hypir/`](hypir/) — one-click HYPIR image restoration/super-resolution: inference, dataset construction, **and** LoRA training on your own images. See its [README](hypir/README.md).
- [`osediff/`](osediff/) — one-click OSEDiff (NeurIPS 2024) one-step diffusion real-world image super-resolution: x4 SR + face restoration inference **and** LoRA fine-tuning (VSD+LPIPS). Single-card 24 GB trainable. See its [README](osediff/README.md).
- [`qwen3vl/`](qwen3vl/) — one-click Qwen3-VL image-to-text (图生文) batch inference: VLM captioning / VQA over a folder of images → per-image `.txt`. See its [README](qwen3vl/README.md).
- [`flux1/`](flux1/) — one-click FLUX.1 text-to-image (文生图) batch inference: prompts (single or per-line file) → PNGs via diffusers `FluxPipeline` (schnell default; dev gated). See its [README](flux1/README.md).
- [`flux2/`](flux2/) — one-click FLUX.2 text-to-image **and** image editing/multi-reference: prompts (+optional ref images) → PNGs via diffusers `Flux2Pipeline`/`Flux2KleinPipeline` (klein-9B default, 4-step distilled; dev 32B/Mistral3 for max quality). See its [README](flux2/README.md).
- [`face_crop/`](face_crop/) — one-click batch face detection & cropping (MediaPipe BlazeFace): multi-face per image → `<stem>_faceN.jpg` sorted left-to-right, resumable CSV log. See its [README](face_crop/README.md).
- [`retouchformer/`](retouchformer/) — one-click RetouchFormer (AAAI 2024) face retouching batch inference: folder of face images → retouched 512×512 PNGs (GPEN + VRT selective self-attention, single forward; Baidu-only weights). See its [README](retouchformer/README.md).
- [`wan22/`](wan22/) — one-click Wan2.2-TI2V-5B (DiffSynth-Studio) text/image-to-video generation: T2V/I2V inference **and** LoRA fine-tuning on your own video dataset. See its [README](wan22/README.md).
- [`wan22_rotate/`](wan22_rotate/) — orbit images → 360° rotation video: SAM 3D Body picks the front-facing image & segments the person (white bg) → Wan2.2 I2V + LoRA generates a 360° rotation video. See its [README](wan22_rotate/README.md).
- [`pdfgs_human/`](pdfgs_human/) — real orbit photos → robust human 3D-Gaussian recon: SAM2 segments all views (white bg) → Pi3 pose + COLMAP → PDF-GS (CVPR 2026 Findings) progressive distractor filtering solves body micro-motion (breathing/hair/clothing). See its [README](pdfgs_human/README.md).
- [`wan_animate_2/`](wan_animate_2/) — one-click Wan-Animate-2 end-to-end character animation: reference image + driving video + prompt → animated video (Base 40-step & Distillation 10-step; multi-GPU sp/sharding auto-patched). See its [README](wan_animate_2/README.md).
- [`pi3_3dgs/`](pi3_3dgs/) — video → 3D reconstruction: Pi3 (π³, ICLR 2026) feed-forward pose+point estimation → COLMAP export → 2D Gaussian Splatting (SIGGRAPH 2024) training + mesh. See its [README](pi3_3dgs/README.md).
- [`minimax_h3/`](minimax_h3/) — one-click MiniMax-H3 (H3-Base 768p) omni-modal video+audio reproduction via SGLang Diffusion: serve T2VA/FL2VA/Ref2VA + reproduce the three official 768p cases. See its [README](minimax_h3/README.md).
- [`eva_gaussian/`](eva_gaussian/) — one-click EVA-Gaussian human novel view synthesis: stereo depth estimation → 3D Gaussian prediction → feature splatting + refinement, two-stage training (depthnet pretrain → full EVA-Gaussian) on THuman2.0/THumansit. See its [README](eva_gaussian/README.md).
- [`world_r1/`](world_r1/) — World-R1 (ICML 2026): Flow-GRPO RL post-training of Wan2.1-T2V with 3D constraints (Depth Anything 3 reconstruction + camera trajectory alignment + Qwen3-VL meta-view) + HPSv2 aesthetic reward. Camera-aware noise wrap + periodic dynamic training. See its [README](world_r1/README.md).
- [`flux_human/`](flux_human/) — 图像生成解决人体微动三维重建模糊：SAM 3D Body 抽 SMPL + 骨骼深度图 → Flux1-dev ControlNet(depth) 生成多视角静止图像 → 无模糊重建（3DGS/NeuS 重建 + 评估为待办；视频生成扩展预留）。See its [README](flux_human/README.md).
- [`artifixer/`](artifixer/) — one-click ArtiFixer (SIGGRAPH 2026) 3D reconstruction enhancement: COLMAP/DL3DV → 3DGRUT sparse recon → ArtiFixer diffusion correction → ArtiFixer3D distillation → ArtiFixer3D+ re-correction. 1.3B variant (fits on single 80 GB A100). See its [README](artifixer/README.md).
- [`4danyone/`](4danyone/) — one-click 4DAnyone (ant-research) monocular video → synchronized multi-view target videos (for 4DGS recon): GVHMR motion recovery → Wan2.2 generator renders N target views. ⚠️ needs ~43 GiB VRAM (server-only, ≥48 GB GPU). See its [README](4danyone/README.md).
