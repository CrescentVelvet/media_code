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
