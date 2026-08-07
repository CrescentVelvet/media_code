# Wan2.2 Rotate 鈥?鐜粫浜虹墿 鈫?姝ｉ潰閫夊浘 鈫?鍒嗗壊 鈫?360掳 鏃嬭浆瑙嗛

浠庝竴缁勭幆缁曚汉鐗╂媿鎽勭殑鍥惧儚涓紝鑷姩鎸戦€変汉鐗╅潰鍚戠浉鏈虹殑姝ｉ潰鍥惧儚锛岀敤 SAM 3D Body 璇嗗埆骞跺垎鍓蹭汉鐗╋紙鑳屾櫙缃櫧锛夛紝鍐嶇敤 Wan2.2-TI2V-5B + 宸茶缁冪殑 LoRA 鐢熸垚 360掳 鏃嬭浆瑙嗛銆?
鏈洰褰曞彧鍚紪鎺掕剼鏈€斺€擲AM 3D Body 瀹樻柟浠ｇ爜鍦?`../sam-3d-body`銆丏iffSynth-Studio 鍦?`../DiffSynth-Studio`锛屾潈閲嶅湪鍚勭畻娉曠殑 `$MODEL_DIR` 涓嬨€備袱姝ュ叡鐢ㄥ悓涓€涓?conda env锛堜粠 `doll` 鍏嬮殕锛岃涓ゅ渚濊禆锛夈€?
## 甯哥敤鍛戒护

> 鍋囪宸茶繘鍏ュ鍣紙鑴氭湰鑷姩婵€娲?`wan22_rotate` env锛夛紱`GPU=0` 鎸夐渶鎹㈠崱銆傞娆¤窇鍓嶅厛鍋氫笅鏂广€岄娆″噯澶囥€嶃€?
```bash
# 鈹€鈹€ 涓€閿細閫夊浘+鍒嗗壊 鈫?鐢熸垚瑙嗛 鈹€鈹€
# INPUT_DIR: 鍚?image/ 瀛愭枃浠跺す鐨勪汉鐗╂暟鎹?# WEIGHT_PATH: 璁粌濂界殑 LoRA
# OUTPUT_DIR: 鍙€夛紝榛樿 ../wan22_rotate_results
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_2d96a34a9df848b2ba80b194df0ae99b \
  WEIGHT_PATH=../../model/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors \
  bash wan22_rotate/run_all.sh

# 鈹€鈹€ 鍒嗘 鈹€鈹€
# 1a) 閫夊浘+鍒嗗壊 瀹屾暣鐗堬紙SAM 3D Body: 3D 濮挎€佷及璁￠€夋闈㈠浘 + SAM 鍒嗗壊锛?GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_2d96a34a9df848b2ba80b194df0ae99b \
  OUTPUT_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01_pick_and_segment.sh
# 1b) 閫夊浘+鍒嗗壊 绠€鍖栫増锛堝彧 ViTDet 妫€娴?+ SAM 鍒嗗壊, 鎸変汉鐗╅潰绉渶澶ч€夊浘, 涓嶅姞杞?3D body 妯″瀷, 鏇村揩锛?GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_2d96a34a9df848b2ba80b194df0ae99b \
  OUTPUT_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/01b_pick_and_segment.sh

# 2) 鍙仛瑙嗛鐢熸垚锛堢敤涓婁竴姝ョ殑鍒嗗壊鍥撅級
GPU=0 WEIGHT_PATH=../../model/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors \
  OUTPUT_DIR=../../output/wan22_rotate_results \
  bash wan22_rotate/02_generate_video.sh

# 鈹€鈹€ 鑷畾涔?鈹€鈹€
# 鎹?prompt / 鍒嗚鲸鐜?/ 甯ф暟锛坧ortrait 榛樿 1248脳704锛沴andscape 鐢?704脳1248锛?GPU=0 INPUT_DIR=... WEIGHT_PATH=... \
  PROMPT="浜虹墿360搴︽棆杞睍绀猴紝楂樿川閲忋€? \
  HEIGHT=1248 WIDTH=706 NUM_FRAMES=121 \
  bash wan22_rotate/run_all.sh
# 璺宠繃閫夊浘姝ラ锛岀洿鎺ョ敤宸叉湁鍥剧墖鐢熸垚瑙嗛
GPU=0 SKIP_SEGMENT=1 \
  SEGMENTED_IMAGE=/path/to/image.png \
  WEIGHT_PATH=... bash wan22_rotate/run_all.sh
# 閫夊嚭鐨勫浘鏄儗闈紵缈昏浆姝ｉ潰鍒ゅ畾鏂瑰悜
GPU=0 FRONTAL_SIGN=-1 INPUT_DIR=... bash wan22_rotate/01_pick_and_segment.sh
# 鐢?SAM2 鍒嗗壊鍣紙闇€鎻愬墠鏀惧ソ sam2 浠撳簱 + checkpoint锛?GPU=0 SEGMENTOR_PATH=/path/to/sam2_repo \
  INPUT_DIR=... bash wan22_rotate/01_pick_and_segment.sh
```

- 缁撴灉锛氬垎鍓插浘 鈫?`../wan22_rotate_results/segmented_image.png`锛涜棰?鈫?`../wan22_rotate_results/rotate_360.mp4`锛涜皟璇曚俊鎭?鈫?`frontal_scores.csv` + `debug_mask.png`銆?
## 棣栨鍑嗗

鏈祦绋嬪缓涓€浠界嫭绔嬬殑 `wan22_rotate` env锛?*CPython 3.10**锛屽尮閰嶆湰鍦?cp310 torch/triton 杞瓙鈥斺€斾笉瑕佺敤 3.11 鎴?clone doll锛宑p310 杞瓙瑁呬笉杩?3.11锛夛紝鎶?sam_3d_body + diffsynth 涓ゅ渚濊禆瑁呭湪涓€璧凤紙detectron2 鐢?`--no-deps` 瑁咃紝`networkx==3.2.1` 瀵?diffsynth 鏃犲奖鍝嶏紱gcc12 鐢?`conda install --no-update-deps` 瑁呴槻 conda 鎶?python 鎺夊寘鎴?GraalPy锛屽惁鍒?numpy 鍏ㄥ潖锛夈€?
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 濉?http_proxy / https_proxy
# 鈿狅笍 纭 proxy.env 涓?http_proxy / https_proxy 涓よ宸插彇娑堟敞閲婂苟濉ソ鍦板潃锛?#    鍚﹀垯 pip 瑁呬緷璧栦細鎶?"Network is unreachable"

# 1. 寤?wan22_rotate env锛堚殸锔?CPython 3.10锛屽尮閰嶆湰鍦?cp310 torch/triton 杞瓙锛?#    涓嶈鐢?3.11 鎴?clone doll鈥斺€攃p310 杞瓙瑁呬笉杩?3.11锛?conda create -n wan22_rotate python=3.10 -y && conda activate wan22_rotate

# 2. 瑁呬袱濂椾緷璧?+ SAM2 + 楠岃瘉
#    INSTALL_DEPS=1 浼氱敤鏈湴 cp310 杞瓙瑁?torch 2.6.0+cu124 + nvidia 渚濊禆锛?#    瑁?gcc12锛?-no-update-deps 闃?GraalPy 鎺夊寘锛夈€乨iffsynth銆乻am_3d_body銆乨etectron2锛?#    骞?clone SAM2 浠撳簱 + pip install + 涓嬭浇 sam2.1_hiera_large.pt
INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh

# 3. 涓嬫潈閲嶏紙涓よ竟鍚勮嚜鐨勪笅杞借剼鏈級
#    瀹屾暣鐗?01 闇€瑕?SAM 3D Body 鏉冮噸锛圙ATED锛夛紱绠€鍖栫増 01b 涓嶉渶瑕?HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh   # SAM 3D Body锛圙ATED锛岄渶鍏?Request access锛?bash wan22/01_verify_models.sh                           # Wan2.2锛堢‘璁ゆ潈閲嶅湪浣嶏級
```

鏉冮噸闇€宸插湪 `$MODEL_DIR`锛堥粯璁?`../../model`锛屽嵆 `../../model`锛変笅锛?```
$MODEL_DIR/
  Wan2.2-TI2V-5B/                          # DiT + T5 + VAE (wan22 鐢? 01_verify_models.sh 鑷姩寤?Wan-AI 绗﹀彿閾炬帴)
  Wan2.1-T2V-1.3B/                         # tokenizer
  Wan2.2-TI2V-5B_lora_add_data_reload/     # 璁粌濂界殑 LoRA
    step-66900.safetensors
  sam-3d-body/
    sam-3d-body-dinov3/                    # SAM 3D Body ckpt + mhr_model
    moge-2-vitl-normal/                    # MoGe2 FOV estimator
```

LoRA 鏉冮噸璺緞绀轰緥锛歚$MODEL_DIR/Wan2.2-TI2V-5B_lora_add_data_reload/step-66900.safetensors`銆?
---

浠ヤ笅涓鸿缁嗗弬鑰冿紙娴佺▼鍘熺悊 / 鍚勬楠ゅ弬鏁?/ 鎺掗敊 / 鐩綍甯冨眬锛夈€?
## Pipeline锛堟祦绋嬭瑙ｏ級

```
INPUT_DIR/image/               (鐜粫浜虹墿鎷嶆憚鐨勫寮犲浘鍍?
    鈹?    鈻?[01] SAM 3D Body  (sam_3d_body env)
    鈹? 鈹溾攢 ViTDet 浜轰綋妫€娴?         鈫?bbox per image
    鈹? 鈹溾攢 MoGe2 FOV 浼拌           鈫?鐩告満鍐呭弬
    鈹? 鈹溾攢 DINOv3 缂栫爜 + MHR 瑙ｇ爜   鈫?3D body (global_rot, pred_vertices, ...)
    鈹? 鈹溾攢 姝ｉ潰璇勫垎 (global_rot)     鈫?閫夋渶浣虫闈㈠浘
    鈹? 鈹斺攢 浜虹墿鍒嗗壊                  鈫?鑳屾櫙缃櫧
    鈹?      鈹溾攢 浼樺厛锛欻umanSegmentor (SAM2/SAM3, 闇€閰?SEGMENTOR_PATH)
    鈹?      鈹溾攢 澶囬€夛細3D mesh silhouette (pyrender 娓叉煋 pred_vertices)
    鈹?      鈹斺攢 鍏滃簳锛歜box 鐭╁舰鎺╃爜
    鈻?$RESULTS_DIR/segmented_image.png   (浜虹墿淇濈暀, 鑳屾櫙鐧借壊)
    鈹?    鈻?[02] Wan2.2-TI2V-5B I2V + LoRA  (wan22 env)
    鈹? 鈹溾攢 鍔犺浇 pipeline (DiT + T5 + VAE, 浠庢湰鍦?$MODEL_DIR)
    鈹? 鈹溾攢 鍔犺浇 LoRA (pipe.load_lora, alpha=1)
    鈹? 鈹斺攢 I2V 鐢熸垚 (鍒嗗壊鍥句綔棣栧抚 鈫?360掳 鏃嬭浆瑙嗛)
    鈻?$RESULTS_DIR/rotate_360.mp4
```

### Step 01 鈥?閫夊浘 + 鍒嗗壊 (`01_pick_and_segment.sh` 鈫?`pick_and_segment.py`)

**姝ｉ潰鍥炬寫閫?*锛氬姣忓紶鍥捐窇 SAM 3D Body 鐨?`process_one_image`锛?D 浜轰綋缃戞牸鎭㈠锛夛紝杈撳嚭鍚?`global_rot`锛堝叏灞€鏃嬭浆锛夈€傚皢鍏惰浆涓烘棆杞煩闃?R锛岃绠椾汉鐗╂湞鍚?`forward = R @ [0,0,1]`锛圫MPL 闈欐濮挎€侀潰 +Z锛夈€傛闈㈡湞鐩告満鏃?`forward[2] < 0`锛堟湞 -Z 鍗崇浉鏈烘柟鍚戯級锛岃瘎鍒?`score = -forward[2]`锛屽彇璇勫垎鏈€楂樼殑鍥俱€?
> 濡傛灉閫夊嚭鏉ョ殑鏄儗闈㈣€屼笉鏄闈紝璇存槑鏃嬭浆绾﹀畾鐩稿弽鈥斺€旇 `FRONTAL_SIGN=-1` 缈昏浆銆?
`global_rot` 鏀寔 3脳3 鏃嬭浆鐭╅樀銆佽酱瑙?(3,)銆佸洓鍏冩暟 (4,) 涓夌鏍煎紡锛岃嚜鍔ㄨ瘑鍒€?
**浜虹墿鍒嗗壊**锛氭寜浠ヤ笅浼樺厛绾у皾璇曪紝浠讳竴鎴愬姛鍗崇敤锛?1. **HumanSegmentor**锛圫AM2/SAM3锛夆€?鍍忕礌绾х簿纭帺鐮併€傞渶 `SEGMENTOR_PATH` 鎸囧悜 sam2 浠撳簱锛堝惈 `checkpoints/` + `configs/`锛夈€傝剼鏈皾璇?`__call__` 鍜?`predict` 涓ょ鎺ュ彛銆?2. **3D mesh silhouette** 鈥?鐢?`Renderer.vertices_to_trimesh` 鏋勫缓 trimesh锛宲yrender 绂诲睆娓叉煋涓轰簩鍊兼帺鐮併€傛棤闇€棰濆閰嶇疆锛屼絾鍙鐩栬韩浣撴ā鍨嬪舰鐘讹紙鍙兘婕忔帀澶村彂銆佸鏉捐。鐗╋級銆?3. **bbox 鐭╁舰** 鈥?鏈€绮楃暐鐨勫厹搴曟帺鐮侊紙甯?`PADDING` 杈硅窛锛夈€?
鎺╃爜搴旂敤锛氫汉鐗╁尯鍩熶繚鐣欏師鍍忕礌锛岃儗鏅尯鍩熻涓虹櫧鑹?`(255,255,255)`锛坄WHITE_BG=0` 鏀逛负榛戣壊锛夈€?
### Step 02 鈥?瑙嗛鐢熸垚 (`02_generate_video.sh`)

鐩存帴璋冪敤宸叉湁鐨?`wan22/02_run_inference.sh`锛堝畠鑷繁 source `wan22/_env.sh` 婵€娲?`wan22` env锛夛紝浼犲叆锛?- `INPUT_IMAGE=$RESULTS_DIR/segmented_image.png`锛圛2V 妯″紡锛?- `WEIGHT_PATH=<lora>.safetensors`锛坄pipe.load_lora(dit, weight, alpha=1)`锛?- `PROMPT`銆乣HEIGHT`/`WIDTH`/`NUM_FRAMES` 绛?
榛樿 portrait锛?248脳704锛夛紝121 甯?@ 15fps 鈮?8 绉掞紝瓒冲涓€鍦?360掳 鏃嬭浆銆傛洿澶氬弬鏁拌 `wan22/README.md` 鐨?inference 閮ㄥ垎銆?
## Config (env vars, all optional)

### Paths & envs
| var | default | note |
| --- | --- | --- |
| `INPUT_DIR` | _(required)_ | 浜虹墿鏂囦欢澶癸紙鍚?`image/` 瀛愭枃浠跺す锛?|
| `WEIGHT_PATH` | _(required for 02)_ | 璁粌濂界殑 LoRA `.safetensors` |
| `GPU` | _(unset)_ | physical GPU id锛宔.g. `GPU=0` |
| `CONDA_ENV` | `wan22_rotate` | conda env锛堜粠 doll 鍏嬮殕锛岃涓ゅ渚濊禆锛?|
| `SAM3D_DIR` | `../sam-3d-body` | SAM 3D Body 瀹樻柟浠ｇ爜 |
| `SAM3D_MODEL_DIR` | `../../model/sam-3d-body` | SAM 3D Body 鏉冮噸 |
| `DIFFSYNTH_DIR` | `../DiffSynth-Studio` | DiffSynth-Studio 浠ｇ爜 |
| `WAN_MODEL_DIR` | `../../model` | Wan2.2 鏉冮噸鏍?|
| `RESULTS_DIR` | `../wan22_rotate_results` | 杈撳嚭鐩綍 |

### Step 01 params
| var | default | note |
| --- | --- | --- |
| `HF_REPO_ID` | `facebook/sam-3d-body-dinov3` | SAM 3D Body 楠ㄥ共锛坅lt `facebook/sam-3d-body-vith`锛?|
| `CHECKPOINT_PATH` | `$SAM3D_MODEL_DIR/<repo>/model.ckpt` | SAM 3D Body checkpoint |
| `MHR_PATH` | `$SAM3D_MODEL_DIR/<repo>/assets/mhr_model.pt` | MHR asset |
| `DEVICE` | `cuda` | falls back to `cpu` if CUDA unavailable |
| `DETECTOR_NAME` | `vitdet` | `vitdet` \| `sam3` \| `` (disable 鈫?full-image bbox) |
| `DETECTOR_PATH` | _(unset)_ | ViTDet auto-downloads; set for offline |
| `SEGMENTOR_NAME` | `sam2` | `sam2` (needs `SEGMENTOR_PATH`) \| `sam3` \| `` (disable) |
| `SEGMENTOR_PATH` | _(unset)_ | sam2 repo dir w/ `checkpoints/` + `configs/` |
| `FOV_NAME` | `moge2` | `moge2` \| `` (disable 鈫?default FOV) |
| `FOV_PATH` | `$SAM3D_MODEL_DIR/moge-2-vitl-normal` | MoGe2 local dir |
| `BBOX_THRESH` | `0.8` | detector score threshold |
| `INFERENCE_TYPE` | `body` | `body` (skip hand decoder, faster) \| `full` \| `hand` |
| `FRONTAL_SIGN` | `1` | `-1` = flip front-facing criterion (if picks back-facing) |
| `WHITE_BG` | `1` | `1` = white background; `0` = black |
| `PADDING` | `0.1` | bbox padding ratio (for bbox mask fallback) |

### Step 02 params
| var | default | note |
| --- | --- | --- |
| `PROMPT` | `浜虹墿360搴︽棆杞睍绀猴紝楂樿川閲忥紝缁嗚妭娓呮櫚銆俙 | 鐢熸垚鎻愮ず璇?|
| `SEGMENTED_IMAGE` | `$RESULTS_DIR/segmented_image.png` | I2V 杈撳叆鍥?|
| `HEIGHT` / `WIDTH` | `1248` / `704` | portrait锛坙andscape 鐢?`704`/`1248`锛?|
| `NUM_FRAMES` | `121` | 鐢熸垚甯ф暟锛?k+1锛學an 绾︽潫锛?|
| `FPS` | `15` | 杈撳嚭 mp4 甯х巼 |
| `OUTPUT_NAME` | `rotate_360` | 杈撳嚭鏂囦欢鍚嶏紙涓嶅惈鎵╁睍鍚嶏級 |
| `SKIP_SEGMENT` | `0` | `1` = 璺宠繃 step 01锛坮un_all.sh 鐢級 |
| `LOW_VRAM` | `0` | `1` = 纾佺洏 offload锛堟參浣嗙渷鏄惧瓨锛岃瑙?wan22 README锛?|

## 鍙兘閬囧埌鐨勯棶棰?
**1. 閫夊嚭鐨勫浘鏄儗闈㈣€屼笉鏄闈?*
SAM 3D Body 鐨?`global_rot` 鏃嬭浆绾﹀畾鍙兘涓庝綘鐨勪汉鐗╂暟鎹浉鍙嶃€傝 `FRONTAL_SIGN=-1` 缈昏浆璇勫垎鏂瑰悜锛岄噸璺?step 01銆?
**2. 鍒嗗壊鐢ㄤ簡 bbox 鍏滃簳锛堟帺鐮佹槸鐭╁舰锛?*
璇存槑 HumanSegmentor 鍜?mesh silhouette 閮藉け璐ヤ簡銆傛敼鍠勬柟娉曪細
- 閰嶇疆 SAM2 鍒嗗壊鍣細璁?`SEGMENTOR_PATH=/path/to/sam2_repo`锛堝惈 `checkpoints/sam2.1_hiera_large.pt` + `configs/`锛夈€?- mesh silhouette 澶辫触閫氬父鏄?pyrender OpenGL 闂锛氱‘璁?`PYOPENGL_PLATFORM=egl`锛坄_env.sh` 榛樿璁句簡锛夛紝鎴栬瘯 `PYOPENGL_PLATFORM=osmesa`锛堥渶 `apt install libosmesa6-dev` + 閲嶈 PyOpenGL锛夈€?
**3. step 01 鎶?`import sam_3d_body` / `cv2` 澶辫触**
`wan22_rotate` env 缂?sam_3d_body 渚濊禆銆俙INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh` 閲嶈銆?
**4. step 02 鎶?`import diffsynth` 澶辫触**
`wan22_rotate` env 缂?diffsynth銆俙INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh` 閲嶈銆?
**5. 瑙嗛鐢熸垚 OOM**
- 闄嶅垎杈ㄧ巼 / 甯ф暟锛歚HEIGHT=480 WIDTH=832 NUM_FRAMES=49`銆?- 寮€纾佺洏 offload锛歚LOW_VRAM=1`銆?- 璇﹁ `wan22/README.md` 鎺掗敊 #3銆?
**6. 澶氫汉鐗╁浘鍍?*
鑴氭湰鍙?bbox 闈㈢Н鏈€澶х殑閭ｄ釜浜猴紙绂荤浉鏈烘渶杩戯級銆傚闇€鎸囧畾鍏朵粬浜猴紝淇敼 `pick_and_segment.py` 鐨?`best = max(outputs, ...)` 閫昏緫銆?
**7. 璺?`.sh` 鎶?`syntax error near unexpected token ('`**
CRLF 琛屽熬姹℃煋銆俙find wan22_rotate -name '*.sh' -exec sed -i 's/\r$//' {} +` 鎴?`git checkout -- wan22_rotate/*.sh`锛坄.gitattributes` 寮哄埗 LF锛夈€?
**8. NumPy 鍧忎簡 / `import numpy` 鎶?ABI 涓嶅吋瀹?/ python 鍙樻垚浜?GraalPy**
鏍瑰洜锛歚conda install -c conda-forge gxx_linux-64`锛堣 detectron2 缂栬瘧瑕佺殑 gcc12锛?*涓嶅甫 `--no-update-deps`** 鏃讹紝conda 姹傝В鍣ㄤ細鎶?env 鐨?python 瀹炵幇鎺夊寘鎴?**GraalPy**锛堟弧瓒?python 妲戒綅鐨勫彟涓€涓?conda-forge 鍖咃級锛岃€?numpy/torch 鏄?CPython ABI 缂栬瘧鐨勨€斺€擥raalPy 涓嬪叏鍧忥紱鏈湴 cp310 torch/triton 杞瓙鏇寸洿鎺ヨ涓嶄笂銆傛湰浠?`00_setup_env.sh` 鐨?gxx 姝ラ**宸插姞 `--no-update-deps`** 闃叉帀鍖咃紝env 涔熷己鍒?CPython 3.10锛堝缓瀹屽嵆鏍￠獙 `platform.python_implementation()=="CPython"`锛実xx 瑁呭畬鍐嶆牎楠屼竴娆★級銆傝嫢 env 宸茶鎺夊寘鎴?GraalPy锛?*閲嶅缓鍗冲彲**锛?```bash
python -c "import platform; print(platform.python_implementation())"   # 杈撳嚭 GraalPy 鍗充腑鎷?conda env remove -n wan22_rotate
conda create -n wan22_rotate python=3.10 -y && conda activate wan22_rotate   # CPython 3.10
INSTALL_DEPS=1 bash wan22_rotate/00_setup_env.sh                          # gxx 甯?--no-update-deps
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"  # 楠岃瘉
```
> 閾佸緥锛氬線 `wan22_rotate` env 閲?`conda install` 浠讳綍鍖呴兘鍔?`--no-update-deps`锛屽惁鍒?GraalPy 浼氬洖鏉ユ妸 numpy 骞叉帀銆?
## 鐩綍甯冨眬
```
<code-dir>/
鈹溾攢鈹€ media_code/                  # 鏈粨
鈹?  鈹溾攢鈹€ proxy.env                # 浠ｇ悊 + 瑕嗙洊椤? gitignored
鈹?  鈹溾攢鈹€ wan22/                   # Wan2.2 鎺ㄧ悊/璁粌鑴氭湰 (step 02 璋冪敤)
鈹?  鈹溾攢鈹€ sam_3d_body/             # SAM 3D Body 鎺ㄧ悊鑴氭湰 (step 01 璋冪敤)
鈹?  鈹斺攢鈹€ wan22_rotate/            # 鈫?鏈洰褰曪紙缂栨帓鑴氭湰锛?鈹溾攢鈹€ sam-3d-body/                 # SAM 3D Body 瀹樻柟浠ｇ爜
鈹溾攢鈹€ sam2/                        # SAM2 瀹樻柟浠ｇ爜 + checkpoints (00 clone, 01b 鍒嗗壊鐢?
鈹?  鈹斺攢鈹€ checkpoints/
鈹?      鈹斺攢鈹€ sam2.1_hiera_large.pt
鈹溾攢鈹€ DiffSynth-Studio-Human/     # DiffSynth-Studio 瀹樻柟浠ｇ爜 (鏈祦绋嬩笓鐢? 00 clone)
鈹溾攢鈹€ wan22_experiments/           # LoRA 璁粌浜х墿 (epoch-N.safetensors)
鈹斺攢鈹€ wan22_rotate_results/        # 鏈祦绋嬭緭鍑?    鈹溾攢鈹€ segmented_image.png      #   姝ｉ潰鍥?(浜虹墿淇濈暀, 鑳屾櫙鐧?
    鈹溾攢鈹€ front_facing_original.jpg#   鍘熷姝ｉ潰鍥?    鈹溾攢鈹€ frontal_scores.csv       #   鍚勫浘姝ｉ潰璇勫垎
    鈹溾攢鈹€ debug_mask.png           #   鍒嗗壊鎺╃爜 (璋冭瘯)
    鈹斺攢鈹€ rotate_360.mp4           #   360掳 鏃嬭浆瑙嗛
```

## Notes
- Official code & weights follow their own licenses (Wan2.2 = Apache 2.0; SAM 3D Body = SAM License). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored 鈥?never committed.
