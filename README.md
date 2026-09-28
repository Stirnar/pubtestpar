# Parasites Pipeline

Command-line tools for preparing microscope images, generating parasite bounding-box proposals, and reviewing annotations in a browser.

The repository is organized around a small end-to-end workflow:

1. Index raw images and flatten them into a consistent filename scheme.
2. Optionally tile flattened images into smaller patches.
3. Generate candidate bounding boxes automatically.
4. Review, edit, and export annotations through a local web UI.

## What The Pipeline Does

- `db`: scans local `data/raw` or GCS `raw/`, hashes files, detects duplicates, copies canonicalized images into the matching `preprocess/flat_images`, and writes a CSV index.
- `tile`: extracts overlapping tiles from microscope images while keeping only regions inside the field of view.
- `detect`: creates parasite proposal boxes plus optional visualizations and YOLO labels.
- `review`: launches a Flask review app for approving, rejecting, relabeling, or drawing boxes manually.
- `external-counts`: prints a CSV table of parasite annotation counts for public datasets under `data/external`.
- `external-audit`: reports external dataset files, annotations, image matching, and exclusion reasons.
- `external-convert-xml`: converts one external Pascal VOC XML dataset into a single COCO-style JSON file.
- `external-review`: converts one public external dataset into review-tool proposals and opens the review UI.
- `ml export-labels`: converts review-tool annotations into named YOLO label files.
- `ml preview-annotations`: draws YOLO bounding boxes on copied source images for sanity-check previews.
- `prepare-data`: creates deterministic train/validation/test splits and a YOLO dataset folder.
- `train`: fine-tunes a selected YOLO model with early stopping.
- `train-experiments`: runs a small YOLO augmentation experiment suite.
- `eval`: evaluates a saved YOLO model on the validation or test split.
- `eval-experiments`: evaluates missing YOLO experiment folders from `weights/best.pt`.
- `export-predictions`: exports a portable YOLO prediction viewer bundle.
- `view-yolo`: launches a local browser viewer for an exported prediction bundle.

## Repository Layout

```text
.
├── main.py
├── file_io/
│   ├── flatten_and_index.py
│   └── storage.py
├── preprocessing/
│   └── tiling.py
├── ml/
│   ├── dataset_manifest.py
│   ├── yolo_prediction_bundle.py
│   ├── yolo_experiment_eval.py
│   └── yolo_training.py
├── bounding_boxes/
│   ├── parasite_detector.py
│   └── review_tool.py
└── data/
    ├── raw/
    └── preprocess/
```

## Expected Raw Data Layout

The `db` command expects images under this structure, either locally or in GCS:

```text
data/raw/<microscope>/<parasite>/image_file.ext
gs://stool-parasite-images/raw/<microscope>/<parasite>/image_file.ext
```

Examples:

```text
data/raw/Carson_Microscope/Trichuria_Trichuris/sample_001.tif
gs://stool-parasite-images/raw/Lab Microscope/Trichuria_Trichuris/sample_001.tif
```

The microscope and parasite folder names are used as metadata fields in the generated index and in the canonical flattened filenames. Additional folders below the parasite directory are allowed for organization, but they are not treated as metadata.

## Installation

Use the project environment file as the source of truth:

```bash
conda env create -f environment.yml
conda activate parasites
```

If the environment already exists, update it after dependency changes:

```bash
conda env update -f environment.yml --prune
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

To prepare data and train YOLO:

```bash
python3 main.py ml export-labels ./bounding_boxes/annotations.json

python3 main.py prepare-data \
  --parasite trichuria_trichuris \
  --train-microscope Foldscope \
  --val-microscope Foldscope \
  --test-microscope Lab_Microscope \
  --train-images 120 \
  --val-images 30 \
  --test-images 30 \
  --seed 42 \
  --class-name trichuria_trichuris

python3 main.py train \
  --data ./runs/datasets/seed42_train120_val30_test30/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --patience 20
```

## Commands

### `db`

Build or update the master image index from local `data/raw` or a GCS bucket.

```bash
python3 main.py db
```

To run against the project bucket:

```bash
python3 main.py db --storage-root gs://stool-parasite-images
```

You can also set the storage root once for a shell or cloud job:

```bash
export PARASITES_STORAGE_ROOT=gs://stool-parasite-images
python3 main.py db
```

What it writes:

- local: `data/preprocess/flat_images/` and `data/preprocess/metadata/master_index.csv`
- GCS: `gs://stool-parasite-images/preprocess/flat_images/` and `gs://stool-parasite-images/preprocess/metadata/master_index.csv`

Useful options:

- `--repo-root PATH`: override repo-root discovery
- `--storage-root PATH_OR_GS_URI`: base storage root for `raw`, `preprocess/flat_images`, and `preprocess/metadata/master_index.csv`
- `--raw-root PATH_OR_GS_URI`: override raw image root
- `--flat-out PATH_OR_GS_URI`: override flattened image output root
- `--csv-out PATH_OR_GS_URI`: override master-index CSV output path
- `--rehash-all`: hash every file on every run
- `--keep-duplicates`: keep files with duplicate content hashes

Notes:

- Repo root is auto-discovered by walking upward until `data/raw` is found.
- GCS runs use Application Default Credentials. Locally, run `gcloud auth application-default login`; on GCP, attach a service account with bucket read/write permissions.
- `prepare-data` can read `gs://` inputs directly and writes a local YOLO dataset for training. `train` and `eval` can stage `gs://` datasets or checkpoints locally before running Ultralytics, then optionally upload results back to GCS.
- On GCP, make sure the VM service account has bucket read/write permissions and enough local disk for the staged prepared dataset and YOLO outputs.
- Images are identified by SHA256, so duplicate detection is content-based rather than filename-based.
- Flattened filenames are generated as `<microscope>__<parasite>__<sha12>.<ext>`.
- GCS `raw_path` and `flat_path` values are stored as `gs://` URIs.
- Flattened images are byte-for-byte copies of the source files. EXIF metadata is preserved, and images are not decoded, re-encoded, rotated, or color-adjusted by `db`.
- If older flattened images were already rewritten, run `python3 main.py db --rehash-all` to replace flat copies whose bytes differ from their source files.

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
- `--output-json`: optional COCO proposal JSON path, default `<output-dir>/proposals.json`
- `--min-frac`: minimum object area as a fraction of field-of-view area, default `0.0003`
- `--max-frac`: maximum object area fraction, default `0.20`
- `--nms-iou`: IoU threshold for non-max suppression, default `0.3`
- `--format`: `coco`, `yolo`, or `both` (default)
- `--force`: reprocess all images even if the proposal JSON already exists
- `--no-vis`: skip visualization images

What it writes:

- COCO-style proposal file: `<output-dir>/proposals.json` by default, or the
  exact path passed with `--output-json`
- `<output-dir>/yolo_labels/`: YOLO label files when `--format yolo` or `both`
- `<output-dir>/visualizations/`: rendered detections unless `--no-vis`

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

Candidate generation also works on external flat image folders. For the flat
NAING training folder, save candidate boxes next to the folder:

```bash
python3 main.py detect "./data/external/NAING/1. Original Dataset/FLAT Training (90%)" \
  --output-json "./data/external/NAING/1. Original Dataset/candidate_annotations.json" \
  --format coco \
  --no-vis
```

This writes:

```text
data/external/NAING/1. Original Dataset/candidate_annotations.json
```

For flat external folders whose image names start with the parasite name, such
as `A. lumbricoides (10).jpeg`, candidate labels are inferred from the filename
and normalized with the same external-dataset label rules used by
`external-counts`. Rerunning `detect` on an existing proposal JSON also fills
old missing, `Noise`, or generic `proposal` labels without recomputing boxes.

Review those candidates with:

```bash
python3 main.py review \
  --images "./data/external/NAING/1. Original Dataset/FLAT Training (90%)" \
  --proposals "./data/external/NAING/1. Original Dataset/candidate_annotations.json" \
  --output "./data/external/NAING/1. Original Dataset/reviewed_candidate_annotations.json"
```

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

### `external-counts`

Summarize public datasets under `data/external` as a CSV table. Rows are
datasets and columns are normalized parasite names. Counts are annotation/object
counts, not image counts.

```bash
python3 main.py external-counts \
  --external-root ./data/external
```

Write the table to disk:

```bash
python3 main.py external-counts \
  --external-root ./data/external \
  --output ./runs/external_counts.csv
```

When `--output` is used, the command also writes a normalization dictionary next
to the CSV:

```text
runs/external_counts_label_normalization.json
```

That JSON contains the explicit normalization rules plus every raw label found
in the external annotation files and the canonical name it became. You can pick
a different path, or save only this dictionary while printing the CSV to stdout,
with:

```bash
python3 main.py external-counts \
  --external-root ./data/external \
  --output ./runs/external_counts.csv \
  --normalization-output ./runs/external_label_normalization.json
```

Report counts separately for each subfolder inside a dataset:

```bash
python3 main.py external-counts \
  --external-root ./data/external \
  --by-subfolder \
  --output ./runs/external_counts_by_subfolder.csv
```

Count one dataset while preserving nested subfolders:

```bash
python3 main.py external-counts \
  --external-root ./data/external \
  --dataset AI4NTD-P15V2-P30Eval \
  --by-subfolder \
  --output ./runs/ai4ntd_external_counts_by_subfolder.csv
```

The command supports the external annotation formats currently present in the
repo: COCO JSON, one-JSON-per-image annotation files, Pascal VOC XML, the USB
microscope CSV pair, and YOLO label folders. Label variants and abbreviations
are normalized before counting; examples include `A. lumbricoides`, `ASC.`, and
`al` as `ascaris_lumbricoides`; `T. trichuria`, `TRI.`, and `tt` as
`trichuris_trichiura`; `h` and `HKW.` as `hookworm`; and `SM`, `SCH.`, and
`S. mansoni` as `schistosoma_mansoni`. For YOLO datasets, names from
`data.yaml` are used when available. If there is no class-name metadata, class
names are inferred from filenames when possible; otherwise they are reported as
`class_<id>`. With `--by-subfolder`, each row is
`dataset,subfolder,<parasite counts>`. Subfolders are inferred from image paths
when available, then annotation paths. For COCO JSON, matching is local to the
annotation file and its sibling `images/` directory so same-name files in other
subfolders are not used for grouping. A terminal image container such as
`images/` is omitted from the reported subfolder, and zero-width or zero-height
boxes are not counted.

### `external-audit`

Audit each external dataset folder and report raw files, raw annotations,
annotations kept by the parser, annotations that can be reviewed locally, and
exclusion reasons.

```bash
python3 main.py external-audit \
  --external-root ./data/external \
  --output ./runs/external_dataset_audit.csv
```

Split nested datasets by subfolder instead of aggregating the full top-level
dataset:

```bash
python3 main.py external-audit \
  --external-root ./data/external \
  --by-subfolder \
  --output ./runs/external_dataset_audit_by_subfolder.csv
```

To inspect one nested dataset, keep `--external-root` at the folder that
contains external datasets and pass the dataset name explicitly:

```bash
python3 main.py external-audit \
  --external-root ./data/external \
  --dataset AI4NTD-P15V2-P30Eval \
  --by-subfolder \
  --output ./runs/ai4ntd_external_dataset_audit_by_subfolder.csv
```

For AI4NTD this produces separate rows such as
`ai4ntd-p3-0-ood/p3-0_2`, `ai4ntd-p3-0-ood/p3-0_1/animal`,
`ai4ntd-p1.5v2/train`, and `ai4ntd-p1.5v2/valid`.

The same code can also be run as a standalone script:

```bash
python3 external_dataset_audit.py \
  --external-root ./data/external \
  --dataset AI4NTD-P15V2-P30Eval \
  --by-subfolder \
  --output ./runs/ai4ntd_external_dataset_audit_by_subfolder.csv
```

Important columns:

- `source_annotations`: annotation objects found in the primary annotation files.
- `kept_annotations`: annotations accepted by the current parsing/counting rules.
- `reviewable_annotations`: kept annotations whose image is present locally and has usable dimensions.
- `excluded_missing_image_file`: annotations dropped because the annotation points to a missing image and the format requires that image.
- `not_reviewable_missing_image`: kept annotations whose local image file is missing.
- `secondary_annotations`: annotations in duplicate/generated companion files that were not counted as the primary source.

### `external-convert-xml`

Convert one Pascal VOC XML dataset into a single COCO-style JSON file inside
the dataset folder.

```bash
python3 main.py external-convert-xml \
  --external-root ./data/external \
  --dataset kankanet
```

The helminth XML dataset uses the same conversion path:

```bash
python3 main.py external-convert-xml \
  --external-root ./data/external \
  --dataset helminthfinaldataset
```

Default output:

```text
data/external/<dataset>/annotations.json
```

The converted JSON is then picked up automatically by `external-counts` and
`external-review`. XML conversion skips and reports XML files whose referenced
images cannot be found. If an XML file has stale filename metadata but a sibling
image with the same stem exists, that sibling image is used. External folders
named `Test_samples` are ignored because they contain unlabeled held-out images
in some public datasets.

### `external-review`

Convert one public dataset under `data/external` into the same COCO-style
proposal format used by `review`, then launch the browser review tool.

Use a dataset name from the first column of `external-counts`:

```bash
python3 main.py external-review \
  --dataset kankanet \
  --external-root ./data/external \
  --work-dir ./runs/external_review \
  --port 5052
```

```bash
python3 main.py external-review \
  --dataset helminthfinaldataset \
  --external-root ./data/external \
  --work-dir ./runs/external_review \
  --port 5053
```

If the images are stored outside the repo, point `--images` to the local image
folder:

```bash
python3 main.py external-review \
  --dataset parasites-1s07h \
  --external-root ./data/external \
  --images /path/to/downloaded/parasites-1s07h \
  --work-dir ./runs/external_review
```

COCO annotations can be converted even when image files are missing because the
image dimensions are stored in the JSON. YOLO labels require the local image
files for review conversion because normalized YOLO boxes must be converted back
to pixel coordinates.

### `prepare-data`

Prepare a YOLO dataset directory from the flattened image pool and canonical
named labels.

Single-parasite example:

```bash
python3 main.py prepare-data \
  --parasite ascaris_lumbricoides \
  --class-name ascaris_lumbricoides \
  --train-microscope "Lab Microscope" \
  --val-microscope "Lab Microscope" \
  --test-microscope "Lab Microscope" \
  --train-images 50 \
  --val-images 20 \
  --test-images 20 \
  --seed 42 \
  --output-dir ./runs/datasets/ascaris_lab_seed42_train50_val20_test20
```

Multiple-parasite example:

```bash
python3 main.py prepare-data \
  --parasite ascaris_lumbricoides \
  --parasite trichuria_trichuris \
  --class-name ascaris_lumbricoides \
  --class-name trichuria_trichuris \
  --train-microscope "Lab Microscope" \
  --val-microscope "Lab Microscope" \
  --test-microscope "Lab Microscope" \
  --train-images 100 \
  --val-images 30 \
  --test-images 30 \
  --seed 42 \
  --output-dir ./runs/datasets/ascaris_trichuria_lab_seed42_train100_val30_test30
```

GCP example using the bucket's existing `flat_images` folder:

```bash
python3 main.py prepare-data \
  --storage-root gs://stool-parasite-images \
  --images-dir gs://stool-parasite-images/preprocess/flat_images \
  --parasite ascaris_lumbricoides \
  --class-name ascaris_lumbricoides \
  --train-microscope "Lab Microscope" \
  --val-microscope "Lab Microscope" \
  --test-microscope "Lab Microscope" \
  --train-images 50 \
  --val-images 20 \
  --test-images 20 \
  --seed 42 \
  --output-dir ./runs/datasets/ascaris_gcp_seed42_train50_val20_test20 \
  --upload-dir gs://stool-parasite-images/runs/datasets/ascaris_gcp_seed42_train50_val20_test20
```

If canonical labels are still local, keep GCS images but override labels:

```bash
python3 main.py prepare-data \
  --master-index gs://stool-parasite-images/preprocess/metadata/master_index.csv \
  --images-dir gs://stool-parasite-images/preprocess/flat_images \
  --labels-dir ./data/labels/canonical \
  --parasite ascaris_lumbricoides \
  --class-name ascaris_lumbricoides \
  --train-images 50 \
  --val-images 20 \
  --test-images 20 \
  --seed 42
```

Default inputs:

- master index: `data/preprocess/metadata/master_index.csv`
- flattened images: `data/preprocess/flat_images`
- named YOLO labels: `data/labels/canonical`
- output directory: `runs/datasets/seed<seed>_train<train>_val<val>_test<test>`

With `--storage-root gs://stool-parasite-images`, default inputs become:

- master index: `gs://stool-parasite-images/preprocess/metadata/master_index.csv`
- flattened images: `gs://stool-parasite-images/preprocess/flat_images`
- named YOLO labels: `gs://stool-parasite-images/preprocess/YOLO_labels/canonical`

If you pass `--images-dir gs://stool-parasite-images/preprocess/flat_images`
without `--storage-root`, `prepare-data` infers the bucket root and uses the GCS
master index by default.

If no storage paths are passed and the local default master index is missing,
`prepare-data` falls back to `gs://stool-parasite-images`.

What it writes:

- `splits.csv`: exactly two columns, `filename` and `split`
- `dataset.yaml`: YOLO dataset file
- `images/<split>/`: symlinks to selected local flat images, or downloaded copies when the source is GCS
- `labels/<split>/`: numeric YOLO labels generated from canonical named labels

Validation and test images are selected first. With the same source pool, seed,
validation count, and test count, validation/test stay fixed as `--train-images`
increases. Training images are sampled only from the remaining images.

The master index is used for metadata and filenames only. If `flat_path`,
`raw_path`, or `canonical_name` contains an old local path, `prepare-data` strips
that path down to the basename and resolves the image as `--images-dir/<filename>`.
Rows marked with `status` equal to `duplicate` are ignored and are not eligible
for train, validation, or test splits.
Named YOLO label rows whose class starts with `maybe_`, such as
`maybe_trichuria_trichuris`, are dropped before writing numeric labels. If an
image has only `maybe_*` boxes, that image is skipped.

Parameters:

- `--parasite`: parasite to include; repeat for multiple parasites. If omitted, all parasites in the index are eligible.
- `--class-name`: YOLO class name; repeat in the exact numeric class order you want in `dataset.yaml`.
- `--train-images`, `--val-images`, `--test-images`: exact image counts. In the current minimal implementation, these are total counts across the selected dataset, not per-class counts.
- `--seed`: deterministic split seed.
- `--train-microscope`, `--val-microscope`, `--test-microscope`: eligible microscope for each split; repeat to allow multiple microscopes. If omitted, any microscope is eligible for that split.
- `--master-index`, `--images-dir`, `--labels-dir`: override the default input paths.
- `--storage-root`: base local path or GCS bucket root for default master index, images, and labels.
- `--output-dir`: where to write the local prepared dataset. If a `gs://` URI is passed, the dataset is staged locally and uploaded there.
- `--upload-dir`: optional `gs://` URI where the prepared dataset is uploaded after local staging.
- `--skip-missing-labels`: skip images without a canonical label file instead of failing.
- `--overwrite`: replace an existing output directory.

### `train`

Fine-tune a YOLO model selected by the user. The `--model` value can be any
Ultralytics-supported model name or local checkpoint path.

Example:

```bash
python3 main.py train \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --model yolo11n.pt \
  --output-dir ./runs/yolo \
  --run-name ascaris_yolo11n_seed42 \
  --epochs 100 \
  --patience 20 \
  --imgsz 640 \
  --batch 16 \
  --save-period 5
```

Training uses early stopping through `--patience`. Ultralytics writes results,
curves, checkpoints, `best.pt`, and `last.pt` under `runs/yolo/` by default.
The wrapper also creates a `tensorboard/` folder from Ultralytics `results.csv`
when training metrics are available.

If training fails with `Ultralytics is not installed`, update the environment
from the repo file on that machine:

```bash
conda env update -f environment.yml --prune
conda activate parasites
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

On headless Linux/GCP machines, OpenCV must use the headless wheel. If you see
`ImportError: libGL.so.1`, remove the GUI OpenCV wheel and update again:

```bash
conda activate parasites
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
conda env update -n parasites -f environment.yml --prune
python3 -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)"
```

GCP training from the local prepared dataset, uploading results to the bucket:

```bash
python3 main.py train \
  --data ./runs/datasets/ascaris_gcp_seed42_train50_val20_test20/dataset.yaml \
  --model yolo11n.pt \
  --output-dir ./runs/yolo \
  --run-name ascaris_yolo11n_seed42 \
  --epochs 100 \
  --patience 20 \
  --save-period 5 \
  --upload-dir gs://stool-parasite-images/runs/yolo
```

GCP training from an uploaded prepared dataset:

```bash
python3 main.py train \
  --data gs://stool-parasite-images/runs/datasets/ascaris_gcp_seed42_train50_val20_test20/dataset.yaml \
  --model yolo11n.pt \
  --output-dir ./runs/yolo \
  --run-name ascaris_yolo11n_seed42 \
  --epochs 100 \
  --patience 20 \
  --work-dir ./runs/staging \
  --upload-dir gs://stool-parasite-images/runs/yolo
```

Parameters:

- `--data`: path to the `dataset.yaml` written by `prepare-data`.
- `--model`: YOLO model/version or checkpoint, for example `yolo11n.pt`, `yolov8s.pt`, or `./runs/yolo/previous/weights/best.pt`.
- `--model` may also be a `gs://` checkpoint; it will be downloaded to `--work-dir` before training.
- `--output-dir`: parent folder for training results, default `./runs/yolo`.
- `--run-name`: optional run folder name under `--output-dir`.
- `--upload-dir`: optional `gs://` URI where training results are uploaded.
- `--work-dir`: local staging directory for `gs://` datasets or checkpoints, default `./runs/staging`.
- `--epochs`: maximum number of epochs.
- `--patience`: early-stopping patience in epochs.
- `--imgsz`: training image size.
- `--batch`: batch size.
- `--device`: optional device value understood by Ultralytics, such as `cpu`, `0`, or `mps`.
- `--save-period`: save a checkpoint every N epochs; use a small value if you want frequent checkpoints.
- `--overwrite`: allow Ultralytics to reuse an existing `--run-name`.

### `train-experiments`

Run a YOLO augmentation suite. The default `separate` grid creates one folder
per experiment under `--output-dir`. At `--max-intensity 4`, the separate grid
contains:

- `no_aug`: all YOLO augmentation parameters disabled
- `default`: Ultralytics default behavior
- `translation_i1` to `translation_i4`
- `scale_i1` to `scale_i4`
- `rotation_i1` to `rotation_i4`
- `flip_i1` to `flip_i4`
- `brightness_contrast_i1` to `brightness_contrast_i4`
- `hsv_color_i1` to `hsv_color_i4`

The `aggressive` grid adds stronger single-augmentation runs and combined
out-of-distribution stress profiles. At `--max-intensity 4`, it creates 58
experiments:

- `shear_i1` to `shear_i4`
- `perspective_i1` to `perspective_i4`
- `mosaic_i1` to `mosaic_i4`
- `mixup_i1` to `mixup_i4`
- `ood_color_i1` to `ood_color_i4`
- `ood_geometry_i1` to `ood_geometry_i4`
- `mosaic_mix_i1` to `mosaic_mix_i4`
- `ood_acquisition_i1` to `ood_acquisition_i4`

The `compact` grid is the recommended first pass when compute matters. It runs
the fixed mixed grid below and does not use `--strategy`:

- `no_aug`
- `default`
- `hsv_color_i4`
- `scale_i4`
- `ood_color_i2`
- `ood_color_i4`
- `ood_geometry_i2`
- `ood_geometry_i4`
- `ood_acquisition_i2`
- `ood_acquisition_i3`
- `ood_acquisition_i4`
- `mosaic_mix_i2`
- `mosaic_mix_i4`

Example:

```bash
python3 main.py train-experiments \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --model yolo11n.pt \
  --output-dir ./runs/yolo_experiments/ascaris_seed42 \
  --run-prefix ascaris_yolo11n_seed42 \
  --epochs 100 \
  --patience 20 \
  --imgsz 640 \
  --batch 16 \
  --save-period 5
```

Aggressive search for acquisition/device shift:

```bash
python3 main.py train-experiments \
  --data ./runs/datasets/ascaris_trichuria_lab_seed42_train160_val40_test40/dataset.yaml \
  --model yolo11n.pt \
  --grid-preset aggressive \
  --max-intensity 4 \
  --output-dir ./runs/yolo_experiments/ascaris_trichuria_aggressive \
  --run-prefix ascaris_trichuria_yolo11n_seed42 \
  --epochs 150 \
  --patience 30 \
  --imgsz 640 \
  --batch 16 \
  --save-period 5
```

Compact mixed search:

```bash
python3 main.py train-experiments \
  --data ./runs/datasets/ascaris_trichuria_lab_seed42_train160_val40_test40/dataset.yaml \
  --model yolo11n.pt \
  --grid-preset compact \
  --output-dir ./runs/yolo_experiments/ascaris_trichuria_compact \
  --run-prefix ascaris_trichuria_yolo11n_seed42 \
  --epochs 150 \
  --patience 30 \
  --imgsz 640 \
  --batch 16 \
  --save-period 5
```

Run only a subset:

```bash
python3 main.py train-experiments \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --model yolo11n.pt \
  --strategy rotation \
  --strategy flip \
  --max-intensity 2 \
  --output-dir ./runs/yolo_experiments/ascaris_rotation_flip
```

GCP example:

```bash
python3 main.py train-experiments \
  --data gs://stool-parasite-images/runs/datasets/ascaris_gcp_seed42_train50_val20_test20/dataset.yaml \
  --model yolo11n.pt \
  --output-dir ./runs/yolo_experiments/ascaris_seed42 \
  --run-prefix ascaris_yolo11n_seed42 \
  --work-dir ./runs/staging \
  --upload-dir gs://stool-parasite-images/runs/yolo_experiments/ascaris_seed42
```

The suite writes `experiment_manifest.json` in `--output-dir`. Each experiment
suite also writes `experiment_grid.json` before training starts. Use
`experiment_grid.json` to inspect the planned hyperparameters, and
`experiment_manifest.json` to track completed runs. Each experiment folder also
contains normal Ultralytics outputs, `experiment_config.json`, `results.csv`,
weights, plots, and a `tensorboard/` folder generated from `results.csv`.

To inspect curves:

```bash
tensorboard --logdir ./runs/yolo_experiments
```

Parameters:

- `--grid-preset`: `separate`, `compact`, or `aggressive`; default `separate`.
- `--max-intensity`: highest intensity to run, `1`, `2`, `3`, or `4`; default `3`.
- `--strategy`: repeat to run only selected single-augmentation strategies. Choices are `translation`, `scale`, `rotation`, `flip`, `brightness_contrast`, `hsv_color`, `shear`, `perspective`, `mosaic`, and `mixup`. This is not available with `--grid-preset compact`.
- `--skip-default`: skip the Ultralytics default run.
- `--skip-no-aug`: skip the no-augmentation run.
- `--run-prefix`: optional prefix for every experiment folder.
- Other training parameters match `train`: `--data`, `--model`, `--epochs`, `--patience`, `--imgsz`, `--batch`, `--device`, `--save-period`, `--upload-dir`, `--work-dir`, and `--overwrite`.

For single-augmentation experiments, each strategy run starts from `no_aug` and
enables only that augmentation group. `brightness_contrast` uses YOLO's
brightness/value parameter (`hsv_v`); `hsv_color` uses hue, saturation, and
value together. The intensity 4 values intentionally go beyond common
Ultralytics defaults for color, illumination, translation, scale, flips, and
geometry so the search includes harder acquisition-shift conditions.

### `eval`

Evaluate a saved model on validation or test data.

Example:

```bash
python3 main.py eval \
  --model ./runs/yolo/ascaris_yolo11n_seed42/weights/best.pt \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --split test \
  --output-dir ./runs/eval \
  --run-name ascaris_yolo11n_seed42_test
```

Evaluation outputs are written under `runs/eval/` by default. The standard
Ultralytics metrics, PR curves, and `predictions.json` use the Ultralytics eval
defaults. The extra files `confusion_matrix_conf025.png` and
`confusion_matrix_conf025.csv` are recomputed at `score >= 0.25`; this fixed
threshold is used only for those confusion-matrix artifacts.

To also save a portable visualization bundle during evaluation, add
`--save-viewer-bundle`. This runs Ultralytics prediction on the evaluated split
and saves the exact `Results.orig_img` frames plus pixel boxes from
`Results.boxes.xyxy/conf/cls` under `viewer_bundle/`.

GCP evaluation example:

```bash
python3 main.py eval \
  --model gs://stool-parasite-images/runs/yolo/ascaris_yolo11n_seed42/weights/best.pt \
  --data gs://stool-parasite-images/runs/datasets/ascaris_gcp_seed42_train50_val20_test20/dataset.yaml \
  --split test \
  --output-dir ./runs/eval \
  --run-name ascaris_yolo11n_seed42_test \
  --work-dir ./runs/staging \
  --upload-dir gs://stool-parasite-images/runs/eval \
  --save-viewer-bundle \
  --viewer-min-conf 0.001
```

Parameters:

- `--model`: saved YOLO checkpoint to evaluate, usually `weights/best.pt`.
- `--model` may also be a `gs://` checkpoint; it will be downloaded to `--work-dir` before evaluation.
- `--data`: path to the same `dataset.yaml` used for training. It may be local or `gs://`.
- `--split`: `test` or `val`; default is `test`.
- `--output-dir`: parent folder for evaluation results, default `./runs/eval`.
- `--run-name`: optional run folder name under `--output-dir`.
- `--upload-dir`: optional `gs://` URI where evaluation results are uploaded.
- `--work-dir`: local staging directory for `gs://` datasets or checkpoints, default `./runs/staging`.
- `--imgsz`: evaluation image size.
- `--batch`: evaluation batch size.
- `--device`: optional device value understood by Ultralytics.
- `--overwrite`: allow Ultralytics to reuse an existing `--run-name`.

### `eval-experiments`

Evaluate every completed experiment under `runs/yolo_experiments` that does not
already have a matching folder under `runs/eval`. The command finds each
`weights/best.pt` recursively and preserves the relative experiment layout.

Preview what would run:

```bash
python3 main.py eval-experiments \
  --experiments-dir ./runs/yolo_experiments \
  --eval-dir ./runs/eval \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --split test \
  --dry-run
```

`--dry-run` only prints the evaluations that would be launched; remove it to run
YOLO evaluation.

Run the missing evaluations:

```bash
python3 main.py eval-experiments \
  --experiments-dir ./runs/yolo_experiments \
  --eval-dir ./runs/eval \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --split test \
  --imgsz 640 \
  --batch 16
```

If `--data` is omitted, the command tries to read it from each
`experiment_config.json` first, then from the nearest `experiment_manifest.json`.

### `export-predictions` And `view-yolo`

Use this when YOLO is available remotely or on GCP, but not on your local
machine. `export-predictions` runs Ultralytics `model.predict(...)` on a dataset
split and writes a portable viewer bundle. The bundle contains:

- `images/`: JPEG frames encoded from Ultralytics `Results.orig_img`
- `predictions_viewer.json`: pixel boxes from `Results.boxes.xyxy`, confidence,
  class id, and class label

Because the saved images and boxes come from the same Ultralytics `Results`
object, the local viewer does not need EXIF transforms or `predictions.json`
interpretation.

Run this on the machine that has YOLO:

```bash
python3 main.py export-predictions \
  --model ./runs/yolo/ascaris_yolo11n_seed42/weights/best.pt \
  --data ./runs/datasets/ascaris_lab_seed42_train50_val20_test20/dataset.yaml \
  --split val \
  --output-dir ./runs/prediction_bundles \
  --run-name ascaris_yolo11n_seed42_val \
  --min-conf 0.001 \
  --imgsz 640 \
  --overwrite
```

Then copy/download the bundle locally and launch the viewer:

```bash
python3 main.py view-yolo \
  --bundle ./runs/prediction_bundles/ascaris_yolo11n_seed42_val \
  --conf 0.25
```

Parameters:

`export-predictions`:

- `--model`: path to YOLO weights, usually `weights/best.pt`.
- `--data`: `dataset.yaml` used to locate the split images.
- `--split`: dataset split to export, default `val`; choose `train`, `val`, or `test`.
- `--output-dir`, `--run-name`: output bundle location.
- `--min-conf`: lowest confidence requested from YOLO and cached. Use `0.001` if you want the slider to explore very low-confidence detections.
- `--imgsz`, `--iou`, `--device`: passed to Ultralytics prediction.
- `--upload-dir`, `--work-dir`: optional GCP staging/upload options.

`view-yolo`:

- `--bundle`: directory containing `predictions_viewer.json` and `images/`.
- `--conf`: initial confidence threshold in the browser; the slider can change it interactively.
- `--host`, `--port`, `--no-browser`: local Flask viewer options.

### `ml export-labels`

Convert the review tool's `annotations.json` into one named YOLO `.txt` file
per image. The first field is the normalized parasite name, for example
`trichuria_trichuris`, rather than a numeric class ID.

```bash
python3 main.py ml export-labels ./bounding_boxes/annotations.json
```

Default output:

```text
data/labels/canonical/<image_stem>.txt
```

Example output line:

```text
trichuria_trichuris 0.500000 0.500000 0.200000 0.200000
```

The converter:

- keeps only `approved` boxes
- skips labels such as `noise` and `background`
- normalizes labels to lowercase underscore tokens, for example `Ascaris Lumbricoides` becomes `ascaris_lumbricoides`
- writes empty `.txt` files for images without approved non-noise boxes, unless `--no-empty-files` is passed

For the Trichuria final annotations:

```bash
python3 main.py ml export-labels \
  ./annotations/TrichuriaFINALAnnotations.json
```

The generated labels are saved under:

```text
data/labels/canonical/
```

### `ml preview-annotations`

Draw YOLO labels from `data/labels/canonical` on copies of the first N source
images listed in an annotation JSON. The JSON is used only to find/order source
images; the boxes are always read from canonical label files.

Before drawing, the preview command applies EXIF orientation and rotates swapped
width/height images to match the annotation JSON dimensions. This keeps YOLO
coordinates projected onto the same image orientation used during review.

```bash
python3 main.py ml preview-annotations \
  ./annotations/TrichuriaFINALAnnotations.json \
  --images-dir /path/to/trichuria/images \
  --output-dir ./tests/sanity_output/trichuria_final/box_previews \
  --limit 10
```

If the annotation JSON contains absolute image paths, `--images-dir` is optional.
If the JSON only contains filenames, pass the directory containing those images:

```bash
python3 main.py ml preview-annotations \
  ./bounding_boxes/annotations.json \
  --images-dir ./data/preprocess/flat_images \
  --output-dir ./tests/sanity_output/annotation_previews \
  --limit 10
```

Preview PNG copies and their summary are written to:

```text
tests/sanity_output/trichuria_final/box_previews/
tests/sanity_output/trichuria_final/box_previews_summary.txt
```

### ML Sanity Checks

Run the focused test suite after changing annotation ingestion code:

```bash
python3 -m unittest tests.test_dataset_manifest
```

For the Trichuria final annotations, the label export sanity run produced:

```text
Images seen:         149
Label files written: 149
Boxes written:       269
Boxes skipped:       3919
```

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
- `raw_path`
- `raw_filename`
- `ext`
- `size_bytes`
- `mtime_epoch`
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
          "label": "Trichuria trichuris",
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
python3 main.py tile --parasite-type Trichuria_Trichuris

# 3. Propose boxes on full images
python3 main.py detect ./data/preprocess/flat_images

# 4. Review and export final annotations
python3 main.py review
python3 main.py ml export-labels ./bounding_boxes/annotations.json

# 5. Prepare splits, train, and evaluate
python3 main.py prepare-data \
  --parasite trichuria_trichuris \
  --class-name trichuria_trichuris \
  --train-images 120 \
  --val-images 30 \
  --test-images 30 \
  --seed 42
python3 main.py train --data ./runs/datasets/seed42_train120_val30_test30/dataset.yaml --model yolo11n.pt
python3 main.py eval --model ./runs/yolo/train/weights/best.pt --data ./runs/datasets/seed42_train120_val30_test30/dataset.yaml
```

## Notes

- `detect` expects a flat directory of image files, not nested class folders.
- `tile` also expects flattened filenames in the `microscope__parasite__sha12.ext` format created by `db`.
- The review app serves files locally with Flask and writes annotations incrementally as you work.
- `runs/` is ignored by Git and is the default home for prepared datasets, training curves, checkpoints, and evaluation outputs.
