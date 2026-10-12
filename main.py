import argparse
from pathlib import Path
# from CLIP_test.ClipParasite import run_clip_gcs
from file_io.flatten_and_index import find_repo_root, create_update_db
from ml.annotations_to_named_yolo import export_named_yolo_from_annotations, summarize_export_result
from ml.annotation_previews import render_annotation_previews, summarize_preview_result
from ml.dataset_manifest import prepare_dataset_from_namespace, summarize_prepared_dataset
from ml.yolo_training import (
    AGGRESSIVE_AUGMENTATION_STRATEGIES,
    AUGMENTATION_GRID_PRESETS,
    eval_yolo_from_namespace,
    export_yolo_predictions_from_namespace,
    summarize_yolo_prediction_export,
    summarize_yolo_experiment_suite,
    summarize_yolo_run,
    train_yolo_experiments_from_namespace,
    train_yolo_from_namespace,
)

# Main parser
ap = argparse.ArgumentParser(description="Parasite image processing pipeline.")
ap.add_argument("--repo-root", type=str, default=None, help="Optional override for repo root.")

# Subparsers for different commands
subparsers = ap.add_subparsers(dest='command', required=True, help='Action to run')

# --- 'db' command ---
parser_db = subparsers.add_parser('db', help='Create or update the file index database.')
parser_db.add_argument("--rehash-all", action="store_true", help="Hash all files every run (slower, safest).")
parser_db.add_argument("--keep-duplicates", action="store_true", help="Allow duplicate hashes to be copied/indexed.")
parser_db.add_argument(
    "--storage-root",
    default=None,
    help=(
        "Base storage root for db inputs/outputs. Use a local path or "
        "gs://stool-parasite-images. Defaults to local data/ paths, or "
        "PARASITES_STORAGE_ROOT when set."
    ),
)
parser_db.add_argument(
    "--raw-root",
    default=None,
    help="Override raw image root. Accepts a local path or gs:// URI.",
)
parser_db.add_argument(
    "--flat-out",
    default=None,
    help="Override flattened image output root. Accepts a local path or gs:// URI.",
)
parser_db.add_argument(
    "--csv-out",
    default=None,
    help="Override master_index.csv output path. Accepts a local path or gs:// URI.",
)

# --- 'clip' command ---
parser_clip = subparsers.add_parser('clip', help='Run CLIP-based parasite classification.')


# --- 'tile' command ---
parser_tile = subparsers.add_parser('tile', help='Extract tiles from microscope images.')
parser_tile.add_argument("--source-dir", type=str, default="./data/preprocess/flat_images", help="Source directory for images to be tiled.")
parser_tile.add_argument("--dest-dir", type=str, default="./data/preprocess/tiles", help="Destination directory for created tiles.")
parser_tile.add_argument("--microscope-type", 
                         choices=["Carson_Microscope", "Foldscope", "Lab_Microscope", "Uhandy_Microscope"],
                         help="Filter images by microscope type (from filename).")
parser_tile.add_argument("--parasite-type", 
                         choices=["Ascaris_Lumbricoides", "Balantidium_Coli", "Clonorchis_Sinensis", 
                                  "Diphyllibothrium_Latum", "Dipylidium_Cannum", 
                                  "Echinococcus_Gran", "Endolimax_Nana", 
                                  "Entamoeba_Coli", "Entamoeba_Histolytica", 
                                  "Enterobius_Vermicularis", "Fasciola_Hepatica", "Giardia", 
                                  "Hymenolepis_Nana", "Necator_Americanus", "Schistosoma_Mansoni",
                                  "Taenia_Pisiformis", "Taenia_Saginata", "Taenia_Solium", 
                                  "Trichuria_Trichuris"],
                         default="Entamoeba_Coli",
                         help="Filter images by parasite type (from filename).")
parser_tile.add_argument("--patch-size", type=int, default=1024, help="Size of square tiles to extract.")
parser_tile.add_argument("--stride", type=int, default=256, help="Stride for tile extraction.")
parser_tile.add_argument("--min-fov-fraction", type=float, default=1., help="Min fraction of tile in FOV.")

# --- 'detect' command ---
parser_detect = subparsers.add_parser('detect', help='Run automated bounding box proposal detection.')
parser_detect.add_argument("input_dir", help="Directory containing microscope images")
parser_detect.add_argument("--output-dir", default="./bounding_boxes/detection_output", help="Directory to save detection results.")
parser_detect.add_argument(
    "--output-json",
    default=None,
    help="Optional path for the COCO candidate/proposal JSON. Defaults to <output-dir>/proposals.json.",
)
parser_detect.add_argument("--min-frac", type=float, default=0.0003)
parser_detect.add_argument("--max-frac", type=float, default=0.20)
parser_detect.add_argument("--nms-iou", type=float, default=0.3)
parser_detect.add_argument("--nms-containment", type=float, default=0.0,
                           help="Also merge a box when this fraction of it lies inside a higher-confidence box (0 disables; try 0.8).")
parser_detect.add_argument("--format", choices=["coco","yolo","both"], default="both")
parser_detect.add_argument("--force", action="store_true", help="Reprocess all images even if the output JSON already exists.")
parser_detect.add_argument("--no-vis", action="store_true", help="Skip visualization output")

# --- 'review' command ---
parser_review = subparsers.add_parser('review', help='Launch the annotation review web tool.')
parser_review.add_argument("--images", default='./data/preprocess/flat_images', help="Directory containing microscope images")
parser_review.add_argument("--proposals", default='./bounding_boxes/detection_output/proposals.json', help="proposals.json from parasite_detector.py")
parser_review.add_argument("--annotations", default=None, help="Existing annotations.json to resume (optional)")
parser_review.add_argument("--output", default="./bounding_boxes/annotations.json", help="Where to auto-save annotations (default: ./bounding_boxes/annotations.json)")
parser_review.add_argument("--port", type=int, default=5000)
parser_review.add_argument("--host", default="127.0.0.1")
parser_review.add_argument("--no-browser", action="store_true")

# --- 'external-counts' command ---
parser_external_counts = subparsers.add_parser(
    'external-counts',
    help='Print a CSV table with parasite annotation counts for data/external datasets.',
)
parser_external_counts.add_argument("--external-root", default="./data/external")
parser_external_counts.add_argument("--dataset", default=None, help="Optional dataset folder under --external-root to count.")
parser_external_counts.add_argument("--output", default=None, help="Optional CSV output path. Defaults to stdout.")
parser_external_counts.add_argument(
    "--by-subfolder",
    action="store_true",
    help="Report one row per dataset subfolder instead of one row per dataset.",
)
parser_external_counts.add_argument(
    "--normalization-output",
    default=None,
    help=(
        "Optional JSON output path for raw-label normalization choices. "
        "Defaults to <output>_label_normalization.json when --output is used."
    ),
)

# --- 'external-audit' command ---
parser_external_audit = subparsers.add_parser(
    'external-audit',
    help='Audit external dataset files, annotations, image matching, and exclusions.',
)
parser_external_audit.add_argument("--external-root", default="./data/external")
parser_external_audit.add_argument("--dataset", default=None, help="Optional dataset folder under --external-root to audit.")
parser_external_audit.add_argument("--output", default=None, help="Optional CSV output path. Defaults to stdout.")
parser_external_audit.add_argument(
    "--by-subfolder",
    action="store_true",
    help="Report one row per dataset subfolder instead of one row per dataset.",
)

# --- 'external-convert-xml' command ---
parser_external_convert_xml = subparsers.add_parser(
    'external-convert-xml',
    help='Convert Pascal VOC XML annotations from one data/external dataset to one COCO-style JSON file.',
)
parser_external_convert_xml.add_argument("--external-root", default="./data/external")
parser_external_convert_xml.add_argument("--dataset", required=True, help="Dataset folder name under --external-root.")
parser_external_convert_xml.add_argument(
    "--output",
    default=None,
    help="Optional JSON output path. Defaults to annotations.json inside the dataset folder.",
)

# --- 'external-review' command ---
parser_external_review = subparsers.add_parser(
    'external-review',
    help='Launch the review web tool for one data/external dataset.',
)
parser_external_review.add_argument("--external-root", default="./data/external")
parser_external_review.add_argument("--dataset", required=True, help="Dataset name from external-counts.")
parser_external_review.add_argument(
    "--images",
    default=None,
    help="Optional image root. Defaults to the selected dataset folder under --external-root.",
)
parser_external_review.add_argument(
    "--work-dir",
    default="./runs/external_review",
    help="Where converted proposals and reviewed annotations are written.",
)
parser_external_review.add_argument("--annotations", default=None, help="Existing annotations.json to resume.")
parser_external_review.add_argument("--output", default=None, help="Autosave destination. Defaults under --work-dir.")
parser_external_review.add_argument("--port", type=int, default=5052)
parser_external_review.add_argument("--host", default="127.0.0.1")
parser_external_review.add_argument("--no-browser", action="store_true")

# --- 'prepare-data' command ---
parser_prepare_data = subparsers.add_parser(
    'prepare-data',
    help='Prepare deterministic train/validation/test splits for YOLO.',
)
parser_prepare_data.add_argument(
    "--master-index",
    default=None,
    help="Path to master_index.csv. Accepts local paths or gs:// URIs.",
)
parser_prepare_data.add_argument(
    "--images-dir",
    default=None,
    help="Directory containing flattened images. Accepts local paths or gs:// URIs.",
)
parser_prepare_data.add_argument(
    "--labels-dir",
    default=None,
    help="Directory containing named YOLO .txt label files. Accepts local paths or gs:// URIs.",
)
parser_prepare_data.add_argument(
    "--storage-root",
    default=None,
    help=(
        "Base local path or gs:// bucket root used for default master-index, images-dir, "
        "and labels-dir. For GCP use gs://stool-parasite-images."
    ),
)
parser_prepare_data.add_argument(
    "--output-dir",
    default=None,
    help=(
        "Local prepared dataset directory. If a gs:// URI is passed, the dataset is staged "
        "locally and uploaded there."
    ),
)
parser_prepare_data.add_argument(
    "--upload-dir",
    default=None,
    help="Optional gs:// URI where the prepared dataset is uploaded after local staging.",
)
parser_prepare_data.add_argument(
    "--parasite",
    dest="parasites",
    action="append",
    default=None,
    help="Parasite class to include. Repeat for multiple classes.",
)
parser_prepare_data.add_argument(
    "--train-microscope",
    dest="train_microscopes",
    action="append",
    default=None,
    help="Microscope type eligible for the train split. Repeat for multiple microscopes.",
)
parser_prepare_data.add_argument(
    "--val-microscope",
    dest="val_microscopes",
    action="append",
    default=None,
    help="Microscope type eligible for the validation split. Repeat for multiple microscopes.",
)
parser_prepare_data.add_argument(
    "--test-microscope",
    dest="test_microscopes",
    action="append",
    default=None,
    help="Microscope type eligible for the test split. Repeat for multiple microscopes.",
)
parser_prepare_data.add_argument("--train-images", type=int, required=True, help="Exact number of train images.")
parser_prepare_data.add_argument("--val-images", type=int, required=True, help="Exact number of validation images.")
parser_prepare_data.add_argument("--test-images", type=int, required=True, help="Exact number of test images.")
parser_prepare_data.add_argument("--seed", type=int, required=True, help="Split seed for deterministic sampling.")
parser_prepare_data.add_argument(
    "--class-name",
    dest="class_names",
    action="append",
    required=True,
    help="Class name to write into dataset.yaml. Repeat in dataset class order.",
)
parser_prepare_data.add_argument(
    "--skip-missing-labels",
    action="store_true",
    help="Skip images without YOLO labels instead of failing validation.",
)
parser_prepare_data.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace the output directory if it already exists.",
)

# --- 'train' command ---
parser_train = subparsers.add_parser('train', help='Fine-tune a YOLO model.')
parser_train.add_argument("--data", required=True, help="Path to dataset.yaml from prepare-data.")
parser_train.add_argument("--model", required=True, help="YOLO model/version, for example yolo11n.pt or yolov8s.pt.")
parser_train.add_argument("--output-dir", default="./runs/yolo", help="Directory for YOLO training results.")
parser_train.add_argument("--run-name", default=None, help="Optional run folder name inside --output-dir.")
parser_train.add_argument("--upload-dir", default=None, help="Optional gs:// URI where training results are uploaded.")
parser_train.add_argument("--work-dir", default="./runs/staging", help="Local staging directory for gs:// inputs.")
parser_train.add_argument("--epochs", type=int, default=100)
parser_train.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
parser_train.add_argument("--imgsz", type=int, default=640)
parser_train.add_argument("--batch", type=int, default=16)
parser_train.add_argument("--device", default=None, help="Optional Ultralytics device value, such as cpu, 0, or mps.")
parser_train.add_argument("--save-period", type=int, default=5, help="Save checkpoint every N epochs.")
parser_train.add_argument("--overwrite", action="store_true", help="Allow Ultralytics to reuse --run-name.")

# --- 'train-experiments' command ---
parser_train_experiments = subparsers.add_parser(
    'train-experiments',
    help='Run a YOLO augmentation experiment suite.',
)
parser_train_experiments.add_argument("--data", required=True, help="Path to dataset.yaml from prepare-data.")
parser_train_experiments.add_argument("--model", required=True, help="YOLO model/version, for example yolo11n.pt or yolov8s.pt.")
parser_train_experiments.add_argument("--output-dir", default="./runs/yolo_experiments", help="Directory for experiment results.")
parser_train_experiments.add_argument("--run-prefix", default=None, help="Optional prefix for every experiment folder.")
parser_train_experiments.add_argument("--upload-dir", default=None, help="Optional gs:// URI where experiment results are uploaded.")
parser_train_experiments.add_argument("--work-dir", default="./runs/staging", help="Local staging directory for gs:// inputs.")
parser_train_experiments.add_argument("--epochs", type=int, default=100)
parser_train_experiments.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
parser_train_experiments.add_argument("--imgsz", type=int, default=640)
parser_train_experiments.add_argument("--batch", type=int, default=16)
parser_train_experiments.add_argument("--device", default=None, help="Optional Ultralytics device value, such as cpu, 0, or mps.")
parser_train_experiments.add_argument("--save-period", type=int, default=5, help="Save checkpoint every N epochs.")
parser_train_experiments.add_argument(
    "--grid-preset",
    choices=list(AUGMENTATION_GRID_PRESETS),
    default="separate",
    help="Augmentation grid to run. Use compact for the focused mixed OOD grid.",
)
parser_train_experiments.add_argument(
    "--max-intensity",
    type=int,
    choices=[1, 2, 3, 4],
    default=3,
    help="Highest augmentation intensity to run.",
)
parser_train_experiments.add_argument(
    "--strategy",
    dest="strategies",
    action="append",
    choices=list(AGGRESSIVE_AUGMENTATION_STRATEGIES),
    default=None,
    help=(
        "Augmentation strategy to run. Repeat to run a subset. Defaults to the strategies "
        "for the selected --grid-preset. Not available with --grid-preset compact."
    ),
)
parser_train_experiments.add_argument("--skip-default", action="store_true", help="Skip the Ultralytics default run.")
parser_train_experiments.add_argument("--skip-no-aug", action="store_true", help="Skip the no-augmentation run.")
parser_train_experiments.add_argument("--overwrite", action="store_true", help="Allow experiment folders to already exist.")

# --- 'eval' command ---
parser_eval = subparsers.add_parser('eval', help='Evaluate a saved YOLO model.')
parser_eval.add_argument("--model", required=True, help="Path to a saved model, such as runs/yolo/exp/weights/best.pt.")
parser_eval.add_argument("--data", required=True, help="Path to dataset.yaml from prepare-data.")
parser_eval.add_argument("--split", choices=["val", "test"], default="test", help="Dataset split to evaluate.")
parser_eval.add_argument("--output-dir", default="./runs/eval", help="Directory for evaluation results.")
parser_eval.add_argument("--run-name", default=None, help="Optional run folder name inside --output-dir.")
parser_eval.add_argument("--upload-dir", default=None, help="Optional gs:// URI where evaluation results are uploaded.")
parser_eval.add_argument("--work-dir", default="./runs/staging", help="Local staging directory for gs:// inputs.")
parser_eval.add_argument("--imgsz", type=int, default=640)
parser_eval.add_argument("--batch", type=int, default=16)
parser_eval.add_argument("--device", default=None)
parser_eval.add_argument("--overwrite", action="store_true", help="Allow Ultralytics to reuse --run-name.")
parser_eval.add_argument(
    "--save-viewer-bundle",
    action="store_true",
    help="Also save a local prediction viewer bundle using Ultralytics predict outputs.",
)
parser_eval.add_argument(
    "--viewer-min-conf",
    type=float,
    default=0.001,
    help="Lowest confidence saved in the viewer bundle.",
)
parser_eval.add_argument("--viewer-iou", type=float, default=0.7)

# --- 'eval-experiments' command ---
parser_eval_experiments = subparsers.add_parser(
    'eval-experiments',
    help='Evaluate missing YOLO experiment folders using each weights/best.pt.',
)
parser_eval_experiments.add_argument("--experiments-dir", default="./runs/yolo_experiments")
parser_eval_experiments.add_argument("--eval-dir", default="./runs/eval")
parser_eval_experiments.add_argument(
    "--data",
    default=None,
    help="Path to dataset.yaml. If omitted, the script tries experiment_config.json or experiment_manifest.json.",
)
parser_eval_experiments.add_argument("--split", choices=["val", "test"], default="test")
parser_eval_experiments.add_argument("--work-dir", default="./runs/staging")
parser_eval_experiments.add_argument("--imgsz", type=int, default=640)
parser_eval_experiments.add_argument("--batch", type=int, default=16)
parser_eval_experiments.add_argument("--device", default=None)
parser_eval_experiments.add_argument("--dry-run", action="store_true")

# --- 'export-predictions' command ---
parser_export_predictions = subparsers.add_parser(
    'export-predictions',
    help='Export a portable prediction viewer bundle using Ultralytics predict.',
)
parser_export_predictions.add_argument("--model", required=True, help="Path to YOLO weights, usually weights/best.pt.")
parser_export_predictions.add_argument("--data", required=True, help="Path to dataset.yaml from prepare-data.")
parser_export_predictions.add_argument("--split", choices=["train", "val", "test"], default="val")
parser_export_predictions.add_argument("--output-dir", default="./runs/prediction_bundles")
parser_export_predictions.add_argument("--run-name", default=None)
parser_export_predictions.add_argument("--upload-dir", default=None, help="Optional gs:// URI where the bundle is uploaded.")
parser_export_predictions.add_argument("--work-dir", default="./runs/staging")
parser_export_predictions.add_argument("--imgsz", type=int, default=640)
parser_export_predictions.add_argument("--min-conf", type=float, default=0.001)
parser_export_predictions.add_argument("--iou", type=float, default=0.7)
parser_export_predictions.add_argument("--device", default=None)
parser_export_predictions.add_argument("--overwrite", action="store_true")

# --- 'view-yolo' command ---
parser_view_yolo = subparsers.add_parser(
    'view-yolo',
    help='Launch a local viewer for a portable YOLO prediction bundle.',
)
parser_view_yolo.add_argument("--bundle", required=True, help="Directory containing predictions_viewer.json and images/.")
parser_view_yolo.add_argument(
    "--conf",
    type=float,
    default=0.25,
    help="Initial confidence threshold shown in the viewer.",
)
parser_view_yolo.add_argument("--port", type=int, default=5051)
parser_view_yolo.add_argument("--host", default="127.0.0.1")
parser_view_yolo.add_argument("--no-browser", action="store_true")

# --- 'ml' command group ---
parser_ml = subparsers.add_parser('ml', help='Annotation utilities for machine-learning labels.')
ml_subparsers = parser_ml.add_subparsers(dest='ml_command', required=True, help='ML action to run')

parser_export_labels = ml_subparsers.add_parser(
    'export-labels',
    help='Convert review-tool annotations.json to named YOLO label files.',
)
parser_export_labels.add_argument(
    "annotations",
    help="Path to review-tool annotations.json.",
)
parser_export_labels.add_argument(
    "--output-dir",
    default=None,
    help="Directory for named YOLO .txt labels. Defaults to data/labels/canonical.",
)
parser_export_labels.add_argument(
    "--no-empty-files",
    action="store_true",
    help="Only write label files for images with approved non-noise boxes.",
)

parser_preview_annotations = ml_subparsers.add_parser(
    'preview-annotations',
    help='Draw YOLO annotation boxes on copies of source images for a quick visual sanity check.',
)
parser_preview_annotations.add_argument(
    "annotations",
    help="Path to review-tool annotations.json. Used for image filenames/paths only; boxes are read from data/labels/canonical.",
)
parser_preview_annotations.add_argument(
    "--output-dir",
    default="./tests/sanity_output/annotation_previews",
    help="Directory for copied image previews with boxes drawn.",
)
parser_preview_annotations.add_argument(
    "--images-dir",
    default=None,
    help="Image directory to search when image paths in the annotation JSON are relative or filenames only.",
)
parser_preview_annotations.add_argument(
    "--limit",
    type=int,
    default=10,
    help="Number of images to render from the annotation file.",
)

args = ap.parse_args()

def resolve_repo_root(args):
    if args.repo_root:
        return Path(args.repo_root).expanduser().resolve()
    try:
        return find_repo_root(Path.cwd())
    except RuntimeError:
        return Path.cwd().resolve()

def run_create_update_db(args): 
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else None
    create_update_db(
        repo_root,
        args.rehash_all,
        args.keep_duplicates,
        storage_root=args.storage_root,
        raw_root=args.raw_root,
        flat_out=args.flat_out,
        csv_out=args.csv_out,
    )
    
def run_tiling(args):
    from preprocessing.tiling import process_images

    source_path = Path(args.source_dir).expanduser().resolve()
    dest_path = Path(args.dest_dir).expanduser().resolve()
    
    process_images(
        source_dir=source_path,
        dest_dir=dest_path,
        microscope_type=args.microscope_type,
        parasite_type=args.parasite_type,
        patch_size=args.patch_size,
        stride=args.stride,
        min_fov_fraction=args.min_fov_fraction
    )

def run_prepare_data(args):
    repo_root = resolve_repo_root(args)
    result = prepare_dataset_from_namespace(args, repo_root)
    print(summarize_prepared_dataset(result))

def run_train(args):
    result = train_yolo_from_namespace(args)
    print(summarize_yolo_run(result))

def run_train_experiments(args):
    result = train_yolo_experiments_from_namespace(args)
    print(summarize_yolo_experiment_suite(result))

def run_eval(args):
    result = eval_yolo_from_namespace(args)
    print(summarize_yolo_run(result))

def run_eval_experiments(args):
    from ml.yolo_experiment_eval import eval_missing_experiments_from_namespace, summarize_experiment_eval_batch

    result = eval_missing_experiments_from_namespace(args)
    print(summarize_experiment_eval_batch(result))

def run_export_predictions(args):
    result = export_yolo_predictions_from_namespace(args)
    print(summarize_yolo_prediction_export(result))

def run_view_yolo(args):
    from ml.yolo_prediction_bundle import run_prediction_bundle_viewer

    run_prediction_bundle_viewer(args)

def run_external_counts(args):
    import sys

    from external_datasets import external_counts_csv, external_label_normalization_json

    content = external_counts_csv(Path(args.external_root), by_subfolder=args.by_subfolder, dataset=args.dataset)
    output_path = None
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"External counts CSV: {output_path}")
    else:
        print(content, end="")

    normalization_path = None
    if args.normalization_output:
        normalization_path = Path(args.normalization_output).expanduser()
    elif output_path:
        normalization_path = output_path.with_name(f"{output_path.stem}_label_normalization.json")

    if normalization_path:
        normalization_path.parent.mkdir(parents=True, exist_ok=True)
        normalization_path.write_text(external_label_normalization_json(Path(args.external_root)), encoding="utf-8")
        stream = sys.stdout if output_path else sys.stderr
        print(f"External label normalization JSON: {normalization_path}", file=stream)

def run_external_audit(args):
    from external_dataset_audit import external_dataset_audit_csv

    content = external_dataset_audit_csv(Path(args.external_root), by_subfolder=args.by_subfolder, dataset=args.dataset)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"External dataset audit CSV: {output_path}")
    else:
        print(content, end="")

def run_external_convert_xml(args):
    from external_datasets import convert_external_xml_annotations, summarize_xml_conversion

    result = convert_external_xml_annotations(
        external_root=Path(args.external_root),
        dataset_name=args.dataset,
        output=Path(args.output) if args.output else None,
    )
    print(summarize_xml_conversion(result))

def run_external_review(args):
    import argparse

    from bounding_boxes.review_tool import run_review_tool
    from external_datasets import build_external_review_bundle, summarize_review_bundle

    safe_dataset = args.dataset.replace("/", "__").replace(" ", "_")
    bundle = build_external_review_bundle(
        external_root=Path(args.external_root),
        dataset_name=args.dataset,
        output_dir=Path(args.work_dir) / safe_dataset,
        images_dir=Path(args.images) if args.images else None,
        annotations_path=Path(args.output) if args.output else None,
    )
    print(summarize_review_bundle(bundle))
    run_review_tool(
        argparse.Namespace(
            images=str(bundle.images_dir),
            proposals=str(bundle.proposals_path),
            annotations=args.annotations,
            output=str(bundle.annotations_path),
            port=args.port,
            host=args.host,
            no_browser=args.no_browser,
        )
    )

def run_ml(args):
    repo_root = resolve_repo_root(args)
    if args.ml_command == 'export-labels':
        output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data" / "labels" / "canonical"
        result = export_named_yolo_from_annotations(
            annotations_path=Path(args.annotations),
            output_dir=output_dir,
            include_empty_files=not args.no_empty_files,
        )
        print(summarize_export_result(result))
    elif args.ml_command == 'preview-annotations':
        result = render_annotation_previews(
            annotations_path=Path(args.annotations),
            labels_dir=repo_root / "data" / "labels" / "canonical",
            output_dir=Path(args.output_dir),
            images_dir=Path(args.images_dir) if args.images_dir else None,
            limit=args.limit,
        )
        print(summarize_preview_result(result))
    else:
        print(f"Error: Unknown ml command '{args.ml_command}'")

# Main execution logic
if args.command == 'db':
    run_create_update_db(args)
elif args.command == 'clip':
    from CLIP_test.ClipParasite import run_clip_local, run_clip_gcs
    run_clip_gcs()
elif args.command == 'tile':
    run_tiling(args)
elif args.command == 'detect':
    from bounding_boxes.parasite_detector import run_detection
    run_detection(args)
elif args.command == 'review':
    from bounding_boxes.review_tool import run_review_tool

    run_review_tool(args)
elif args.command == 'external-counts':
    run_external_counts(args)
elif args.command == 'external-audit':
    run_external_audit(args)
elif args.command == 'external-convert-xml':
    run_external_convert_xml(args)
elif args.command == 'external-review':
    run_external_review(args)
elif args.command == 'prepare-data':
    run_prepare_data(args)
elif args.command == 'train':
    run_train(args)
elif args.command == 'train-experiments':
    run_train_experiments(args)
elif args.command == 'eval':
    run_eval(args)
elif args.command == 'eval-experiments':
    run_eval_experiments(args)
elif args.command == 'export-predictions':
    run_export_predictions(args)
elif args.command == 'view-yolo':
    run_view_yolo(args)
elif args.command == 'ml':
    run_ml(args)
else:
    print(f"Error: Unknown command '{args.command}'")
    ap.print_help()
