# face_crop

One-click **batch face detection & cropping** on Windows, powered by
[MediaPipe](https://github.com/google/mediapipe) FaceDetection.

Reads every image from an input directory, detects **all** faces in each (a
single image may yield several crops), and saves each cropped face as a JPG.
Crops are named `<original_stem>_face1.jpg`, `_face2.jpg`, ... sorted
**left-to-right**.

## Install

```powershell
pip install mediapipe opencv-contrib-python numpy pillow
# domestic mirror if PyPI is slow:
# pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple
```

The BlazeFace `.tflite` model is **auto-downloaded** into `models/` on first run
(~225 KB short-range / ~1 MB full-range, from Google's public MediaPipe model
bucket). Put it there manually if your network can't reach `storage.googleapis.com`.

> MediaPipe 0.10.20+ removed the legacy `mp.solutions` API, so this script uses
> the current **Tasks API** (`mediapipe.tasks.python.vision.FaceDetector`).

## Usage (one-click)

```bat
:: process all images in C:\code\target_c -> C:\code\target_c_faces
run_face_crop.bat
```

```powershell
# or directly, with options
python crop_faces.py --input C:\code\target_c --output C:\code\target_c_faces
```

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--input` / `-i` | `C:\code\target_c` | input image directory |
| `--output` / `-o` | `C:\code\target_c_faces` | output directory for cropped JPGs |
| `--model-kind` | `short` | BlazeFace model: `short` (<2 m, close/large faces) / `full` (<5 m, smaller/farther faces) |
| `--model-path` | _(auto)_ | override with a path to your own `.tflite` face detector |
| `--min-confidence` | `0.5` | lower finds more faces but more false positives (try `0.3` for small faces) |
| `--pad` | `0.15` | expand each face box by this fraction on every side (clamped to image) |
| `--min-size` | `24` | skip crops smaller than this (px) on either side |
| `--quality` | `95` | JPEG quality 1-100 |
| `--exts` | `_(all common)_` | comma-separated image extensions to process |
| `--limit` | `0` | process only the first N images (smoke test) |
| `--overwrite` | off | ignore the log and reprocess everything |

## Output

- `<stem>_face1.jpg`, `<stem>_face2.jpg`, ... per input image, sorted left-to-right.
- `faces_log.csv` in the output dir: one row per processed image
  (`filename, n_faces, outputs, status`). Used for **resumable** runs — re-running
  skips files already marked `ok`. Delete the CSV (or the output dir) to reprocess.

## Notes

- Input is read with PIL (reliable for `.tif`); MediaPipe receives RGB arrays.
- If a face is missed, lower `--min-confidence` (e.g. `0.3`) or switch to
  `--model-kind full` (better for small/far faces) / `short` (close-up portraits).
  If too many false positives, raise `--min-confidence`.
- For very small faces in low-resolution images, MediaPipe may under-detect;
  consider upscaling the input first.

---

## Paired HQ/LQ face-crop dataset for super-resolution (`crop_faces_paired.py`)

Builds a **pixel-aligned face SR dataset** from a high-quality RAW folder and a
low-quality 360p `.tif` folder of the **same** images (same stems, same content,
only resolution differs). Output is paired PNGs you can feed to an SR trainer:

```
<out>/hq/<stem>_face1.png   ← GT (high-res, decoded from RAW)
<out>/lq/<stem>_face1.png   ← input (360p, same relative face region)
```

### How pairing works

1. Faces are detected **on the LQ image** (BlazeFace's sweet spot; guarantees the
   LQ crop is meaningful).
2. Each padded/clamped face box is converted to **relative [0,1] coordinates**.
3. The **same relative region** is applied to the decoded HQ image (scaled by the
   per-axis pixel ratio), so HQ and LQ crops depict the identical face at
   different resolutions (~10× scale at full decode).

RAW files (`.CR2/.NEF/.ARW/.RAF/.RW2`) are decoded with `rawpy` (LibRaw) with the
**EXIF orientation applied** (so portrait/landscape match the LQ). Images whose
HQ/LQ aspect ratios disagree (a handful of 4:3 Panasonic `.RW2` vs the 2:3 LQ)
are **skipped** to keep every pair geometrically consistent.

### Install (extra dep)

```powershell
pip install rawpy
```

> On a clean Windows without the MSVC runtime, rawpy may fail to load
> (`MSVCP140.dll` missing). Fix: copy `msvcp140.dll` (e.g. from
> `C:\Windows\WinSxS\amd64_microsoft-edge-webview_*`) next to
> `…\site-packages\rawpy\raw_r.dll`, or install the
> [VC++ 2015-2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### Usage

```bat
:: one-click: HQ raw -> C:\code\ppr10k_faces\{hq,lq} paired PNGs
run_face_crop_paired.bat
```

```powershell
python crop_faces_paired.py --workers 12
# faster, ~5x scale:        python crop_faces_paired.py --half-size
# smoke test (first 50):    python crop_faces_paired.py --limit 50 --workers 4
```

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--hq` | `…\raw_zips\raw` | HQ RAW directory (CR2/NEF/ARW/RAF/RW2) |
| `--lq` | `…\train_val_images_tif_360p\source` | LQ `.tif` directory (same stems) |
| `--output` / `-o` | `C:\code\ppr10k_faces` | output root (creates `hq/` and `lq/`) |
| `--half-size` | off | decode HQ at half resolution (~2× faster, ~5× scale; default full ~10.7×) |
| `--workers` | `min(cpu,8)` | process-pool size (RAW decode is the bottleneck) |
| `--model-kind` / `--min-confidence` / `--pad` / `--min-size` | `short`/`0.5`/`0.15`/`24` | same as the single-folder script |
| `--png-level` | `3` | PNG compress_level (0-9; lower = faster + larger) |
| `--limit` / `--overwrite` | `0` / off | smoke-test cap / ignore log & reprocess |

### Output & log

- `hq/<stem>_faceN.png`, `lq/<stem>_faceN.png` — paired, sorted left-to-right.
- `faces_paired_log.csv`: one row per pair (`stem, n_faces, outputs, status`).
  Resumable — re-run skips stems already `ok`. Rows with `status=skip:aspect_mismatch`
  are HQ/LQ aspect mismatches (skipped, no output).

### Notes

- The LQ crops carry the 360p tif's real (camera-style) degradation, which is
  more realistic than bicubic downsampling for SR training. The HQ/LQ scale is
  ~10.7× at full decode (varies slightly per camera: 10.18×–10.75×); resize to a
  fixed scale during your own dataset prep if your model needs it.
- Multi-process: each worker owns one `FaceDetector` + decodes RAWs sequentially.
  `OMP_NUM_THREADS=1` is set so LibRaw doesn't oversubscribe per worker.
