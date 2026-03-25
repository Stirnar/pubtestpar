# Parasites Pipeline

Command-line tools for preparing microscope images, generating parasite bounding-box proposals, and reviewing annotations in a browser.

The repository is organized around a small end-to-end workflow:

1. Index raw images and flatten them into a consistent filename scheme.
2. Optionally tile flattened images into smaller patches.
3. Generate candidate bounding boxes automatically.
4. Review, edit, and export annotations through a local web UI.

## What The Pipeline Does

- `db`: scans `data/raw`, hashes files, detects duplicates, copies canonicalized images into `data/preprocess/flat_images`, and writes a CSV index.
- `tile`: extracts overlapping tiles from microscope images while keeping only regions inside the field of view.
- `detect`: creates parasite proposal boxes plus optional visualizations and YOLO labels.
- `review`: launches a Flask review app for approving, rejecting, relabeling, or drawing boxes manually.

## Repository Layout

```text
.
├── main.py
├── file_io/
│   └── flatten_and_index.py
├── preprocessing/
│   └── tiling.py
├── bounding_boxes/
│   ├── parasite_detector.py
│   └── review_tool.py
└── data/
    ├── raw/
    └── preprocess/
```

## Expected Raw Data Layout

The `db` command expects images under:

```text
data/raw/<microscope>/<parasite>/<user>/image_file.ext
```

Example:

```text
data/raw/Carson_Microscope/Giardia/user_01/sample_001.tif
```

Those folder names are used as metadata fields in the generated index and in the canonical flattened filenames.

## Installation

Use Python 3 and install the packages used by the scripts:

```bash
pip install flask opencv-python-headless numpy scikit-image
```

## Quick Start

If you already have flattened images in `data/preprocess/flat_images`, you can skip straight to detection:

```bash
python3 main.py detect ./data/preprocess/flat_images
python3 main.py review
```

For the full pipeline from raw images:

```bash
python3 main.py db
python3 main.py tile
python3 main.py detect ./data/preprocess/flat_images
python3 main.py review
```

## Commands

### `db`

Build or update the master image index from `data/raw`.

```bash
python3 main.py db
```

What it writes:

- `data/preprocess/flat_images/`: canonicalized copies of images
- `data/preprocess/metadata/master_index.csv`: image index and processing status

Useful options:

- `--repo-root PATH`: override repo-root discovery
- `--rehash-all`: hash every file on every run
- `--keep-duplicates`: keep files with duplicate content hashes

Notes:

- Repo root is auto-discovered by walking upward until `data/raw` is found.
- Images are identified by SHA256, so duplicate detection is content-based rather than filename-based.

### `tile`

Extract square tiles from flattened images.

```bash
python3 main.py tile \
  --source-dir ./data/preprocess/flat_images \
  --dest-dir ./data/preprocess/tiles
```

Useful options:

- `--microscope-type`: filter by microscope name embedded in the flattened filename
- `--parasite-type`: filter by parasite name embedded in the flattened filename
- `--patch-size`: tile size in pixels, default `1024`
- `--stride`: sliding-window stride, default `256`
- `--min-fov-fraction`: minimum fraction of the tile that must lie inside the field of view, default `1.0`

Important default:

- `--parasite-type` currently defaults to `Entamoeba_Coli`. If you want a different organism, pass it explicitly.

Output:

- Tiles are written under `data/preprocess/tiles/<parasite>/`

### `detect`

Generate candidate parasite bounding boxes for each image in a directory.

```bash
python3 main.py detect ./data/preprocess/flat_images
```

Useful options:

- `--output-dir`: default `./bounding_boxes/detection_output`
- `--min-frac`: minimum object area as a fraction of field-of-view area, default `0.0003`
- `--max-frac`: maximum object area fraction, default `0.20`
- `--nms-iou`: IoU threshold for non-max suppression, default `0.3`
- `--format`: `coco`, `yolo`, or `both` (default)
- `--no-vis`: skip visualization images

What it writes:

- `bounding_boxes/detection_output/proposals.json`: COCO-style proposal file
- `bounding_boxes/detection_output/yolo_labels/`: YOLO label files when `--format yolo` or `both`
- `bounding_boxes/detection_output/visualizations/`: rendered detections unless `--no-vis`

Detection methods combined by the script include:

- stain-color segmentation
- local contrast detection
- edge-density detection
- dark-object detection
- local anomaly detection

Example:

```bash
python3 main.py detect ./data/preprocess/flat_images \
  --output-dir ./bounding_boxes/detection_output \
  --min-frac 0.0001
```

Lower `--min-frac` usually increases recall and false positives. Higher values usually reduce the number of proposals.

### `review`

Launch the annotation review app.

```bash
python3 main.py review
```

Default inputs:

- images: `./data/preprocess/flat_images`
- proposals: `./bounding_boxes/detection_output/proposals.json`
- output annotations: `./bounding_boxes/annotations.json`
- host: `127.0.0.1`
- port: `5000`

Useful options:

- `--images PATH`: image directory to review
- `--proposals PATH`: proposal JSON from `detect`
- `--annotations PATH`: resume from an existing annotation file
- `--output PATH`: autosave destination for annotations
- `--host HOST`
- `--port PORT`
- `--no-browser`: do not auto-open the browser

Resume example:

```bash
python3 main.py review \
  --images ./data/preprocess/flat_images \
  --proposals ./bounding_boxes/detection_output/proposals.json \
  --annotations ./bounding_boxes/annotations.json
```

Inside the UI you can:

- approve or reject proposed boxes
- relabel boxes
- draw manual boxes
- export annotations as JSON, COCO, or YOLO-style data

Keyboard shortcuts:

- `A`: approve current box
- `R`: reject current box
- `D`: toggle draw mode
- `Tab`: jump to next pending box
- `Left` / `Right`: move between images

## Standalone Preprocessing Scripts

### `preprocessing/multi_image_super_resolution.py`

This script is intentionally standalone. It does not modify `main.py` or the rest of the pipeline, and it can be run directly on a folder of repeated views of the same scene.

Example:

```bash
cd preprocessing
python3 multi_image_super_resolution.py ../path/to/images \
  --output-dir ../path/to/images/super_resolution_output \
  --scale 2 \
  --top-k 8 \
  --save-aligned
```

The script is self-contained, so it can be launched directly from inside the `preprocessing/` folder as shown above.

Input parameters:

- `input_dir`: folder containing at least two images of the same sample or field of view. The script assumes these images can be aligned and fused.
- `--output-dir`: destination folder for all generated files. If omitted, the script creates `super_resolution_output` inside `input_dir`.
- `--scale`: upsampling factor for the final fused image. For example, `2` produces an output with roughly double the width and height of the reference image.
- `--motion-model`: image-registration model used during alignment. `translation` allows shifts only, `euclidean` allows shifts plus rotation, and `affine` also allows shear and non-uniform geometric changes.
- `--top-k`: optional limit on how many input images to keep. The script first ranks images by sharpness and then keeps the best `K` before fusion.
- `--ecc-iterations`: maximum number of iterations for ECC registration. Higher values can improve difficult alignments but may run more slowly.
- `--ecc-eps`: convergence tolerance for ECC registration. Smaller values can make optimization stricter.
- `--sharpen-amount`: strength of the final unsharp-mask enhancement applied after fusion.
- `--sharpen-sigma`: blur sigma used inside the unsharp mask. Larger values affect broader structures.
- `--save-aligned`: if passed, the script also saves the aligned low-resolution inputs used during fusion.

Outputs:

- `super_resolved_x{scale}.png`: final fused image
- `resolution_metrics.csv`: no-reference resolution and sharpness metrics for each input image and the fused output
- `super_resolution_summary.json`: summary report including metric improvements and ranking against the input images
- `aligned_inputs/`: optional aligned intermediate images when `--save-aligned` is used

Metric notes:

- The reported values are no-reference proxies for sharpness and detail, including Laplacian variance, Tenengrad, Brenner gradient, edge density, and high-frequency energy.
- These are useful for relative comparison between the fused output and the original images, but they are not a substitute for a true optical resolution benchmark.

## Outputs

### `master_index.csv`

The index produced by `db` includes fields such as:

- `image_id`
- `microscope`
- `parasite`
- `user`
- `raw_path`
- `canonical_name`
- `flat_path`
- `status`

Status values include `new`, `updated`, `duplicate`, `copied_exists`, and `skipped_unchanged`.

### `annotations.json`

The review tool autosaves annotations in this shape:

```json
{
  "version": "1.0",
  "generator": "review_tool.py",
  "images": {
    "example.jpg": {
      "width": 952,
      "height": 1288,
      "annotations": [
        {
          "bbox": [747, 632, 38, 33],
          "label": "Giardia lamblia",
          "status": "approved",
          "method": "purple_stain"
        }
      ]
    }
  }
}
```

## Typical Workflow

```bash
# 1. Build the flattened image set and metadata index
python3 main.py db

# 2. Optionally generate tiles for downstream experiments
python3 main.py tile --parasite-type Giardia

# 3. Propose boxes on full images
python3 main.py detect ./data/preprocess/flat_images

# 4. Review and export final annotations
python3 main.py review
```

## Notes

- `detect` expects a flat directory of image files, not nested class folders.
- `tile` also expects flattened filenames in the `microscope__parasite__...` format created by `db`.
- The review app serves files locally with Flask and writes annotations incrementally as you work.
