from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from file_io.storage import GcsUri, StoragePath, backend_for_locations, is_gcs_uri, parse_gcs_uri
from ml.yolo_augmentation_grid import (
    AGGRESSIVE_AUGMENTATION_STRATEGIES,
    AUGMENTATION_GRID_PRESETS,
    DEFAULT_AUGMENTATION_STRATEGIES,
    NO_AUGMENTATION_PARAMS,
    YoloExperiment,
    build_yolo_experiment_plan,
    default_strategies_for_preset,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class YoloRunResult:
    output_dir: Path | None
    model_path: Path | None
    uploaded_dir: str | None = None
    confusion_matrix_path: Path | None = None
    prediction_bundle_path: Path | None = None


@dataclass(frozen=True)
class YoloExperimentSuiteResult:
    output_dir: Path
    grid_path: Path
    manifest_path: Path
    total_experiments: int
    uploaded_dir: str | None = None


@dataclass(frozen=True)
class YoloPredictionExportResult:
    output_dir: Path
    predictions_path: Path
    total_images: int
    total_boxes: int
    uploaded_dir: str | None = None


@dataclass(frozen=True)
class _EvalBox:
    class_id: int
    xyxy: tuple[float, float, float, float]


def train_yolo_from_namespace(args: Any) -> YoloRunResult:
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_path = _stage_dataset_yaml(args.data, work_dir)
    model_path = _stage_model(args.model, work_dir)
    project, upload_root = _local_project_and_upload_root(args.output_dir, args.upload_dir, work_dir / "yolo")
    kwargs = _training_kwargs(args, data_path, project, name=args.run_name)
    model = _load_yolo(model_path)
    result = model.train(**kwargs)
    if getattr(result, "save_dir", None):
        _write_tensorboard_scalars(Path(result.save_dir))
    uploaded_dir = _upload_run_dir(getattr(result, "save_dir", None), upload_root) if upload_root else None
    return _result_paths(result, uploaded_dir)


def train_yolo_experiments_from_namespace(args: Any) -> YoloExperimentSuiteResult:
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_path = _stage_dataset_yaml(args.data, work_dir)
    model_path = _stage_model(args.model, work_dir)
    project, upload_root = _local_project_and_upload_root(
        args.output_dir,
        args.upload_dir,
        work_dir / "yolo_experiments",
    )
    project = project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    grid_preset = args.grid_preset
    strategies = args.strategies or default_strategies_for_preset(grid_preset)
    experiments = build_yolo_experiment_plan(
        max_intensity=args.max_intensity,
        strategies=strategies,
        include_default=not args.skip_default,
        include_no_augmentation=not args.skip_no_aug,
        run_prefix=args.run_prefix,
        grid_preset=grid_preset,
    )
    _check_experiment_folders(project, experiments, overwrite=args.overwrite)

    plan = [asdict(experiment) for experiment in experiments]
    training_metadata = _base_training_metadata(args)
    grid_path = project / "experiment_grid.json"
    grid_document: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "data": str(data_path),
        "model": str(model_path),
        "output_dir": str(project),
        "upload_dir": upload_root,
        "run_prefix": args.run_prefix,
        "grid_preset": grid_preset,
        "strategies": list(strategies),
        "include_default": not args.skip_default,
        "include_no_augmentation": not args.skip_no_aug,
        "max_intensity": args.max_intensity,
        "total_experiments": len(experiments),
        "training": training_metadata,
        "experiments": plan,
    }
    _write_manifest(grid_path, grid_document, upload_root)

    manifest_path = project / "experiment_manifest.json"
    manifest: dict[str, Any] = {
        "created_at_utc": _utc_now(),
        "data": str(data_path),
        "model": str(model_path),
        "output_dir": str(project),
        "upload_dir": upload_root,
        "grid_path": str(grid_path),
        "grid_preset": grid_preset,
        "strategies": list(strategies),
        "max_intensity": args.max_intensity,
        "training": training_metadata,
        "plan": plan,
        "experiments": [],
    }
    _write_manifest(manifest_path, manifest, upload_root)

    for experiment in experiments:
        kwargs = _training_kwargs(args, data_path, project, name=experiment.name)
        kwargs.update(experiment.augmentation)
        model = _load_yolo(model_path)
        result = model.train(**kwargs)
        save_dir = Path(result.save_dir)
        _write_json(
            save_dir / "experiment_config.json",
            {
                "experiment": asdict(experiment),
                "training": training_metadata,
                "yolo_args": kwargs,
            },
        )
        tensorboard_dir = _write_tensorboard_scalars(save_dir)
        uploaded_dir = _upload_run_dir(save_dir, upload_root) if upload_root else None
        yolo_result = _result_paths(result, uploaded_dir)
        manifest["experiments"].append(
            {
                **asdict(experiment),
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "output_dir": str(yolo_result.output_dir) if yolo_result.output_dir else None,
                "model_path": str(yolo_result.model_path) if yolo_result.model_path else None,
                "uploaded_dir": uploaded_dir,
                "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir else None,
            }
        )
        _write_manifest(manifest_path, manifest, upload_root)

    return YoloExperimentSuiteResult(
        output_dir=project,
        grid_path=grid_path,
        manifest_path=manifest_path,
        total_experiments=len(experiments),
        uploaded_dir=upload_root,
    )


def eval_yolo_from_namespace(args: Any) -> YoloRunResult:
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_path = _stage_dataset_yaml(args.data, work_dir)
    model_path = _stage_model(args.model, work_dir)
    project, upload_root = _local_project_and_upload_root(args.output_dir, args.upload_dir, work_dir / "eval")
    kwargs: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project),
        "plots": True,
        "save_json": True,
        "exist_ok": args.overwrite,
    }
    if args.run_name:
        kwargs["name"] = args.run_name
    if args.device:
        kwargs["device"] = args.device

    model = _load_yolo(model_path)
    result = model.val(**kwargs)
    save_dir = Path(result.save_dir)
    confusion_matrix_path = _write_confidence_confusion_matrix(
        dataset_yaml=data_path,
        split=args.split,
        predictions_json=Path(result.save_dir) / "predictions.json",
        output_dir=save_dir,
        confidence=0.25,
    )
    prediction_bundle_path = None
    if getattr(args, "save_viewer_bundle", False):
        bundle_result = _write_prediction_bundle_for_split(
            model=model,
            model_path=str(model_path),
            dataset_yaml=data_path,
            split=args.split,
            output_dir=save_dir / "viewer_bundle",
            imgsz=args.imgsz,
            min_conf=args.viewer_min_conf,
            iou=args.viewer_iou,
            device=args.device,
            overwrite=True,
        )
        prediction_bundle_path = bundle_result.output_dir
    uploaded_dir = _upload_run_dir(getattr(result, "save_dir", None), upload_root) if upload_root else None
    return _result_paths(
        result,
        uploaded_dir,
        confusion_matrix_path=confusion_matrix_path,
        prediction_bundle_path=prediction_bundle_path,
    )


def export_yolo_predictions_from_namespace(args: Any) -> YoloPredictionExportResult:
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_path = _stage_dataset_yaml(args.data, work_dir)
    model_path = _stage_model(args.model, work_dir)
    project, upload_root = _local_project_and_upload_root(
        args.output_dir,
        args.upload_dir,
        work_dir / "prediction_bundles",
    )
    run_name = args.run_name or f"predictions_{args.split}"
    output_dir = project.expanduser().resolve() / run_name

    model = _load_yolo(model_path)
    bundle_result = _write_prediction_bundle_for_split(
        model=model,
        model_path=str(model_path),
        dataset_yaml=data_path,
        split=args.split,
        output_dir=output_dir,
        imgsz=args.imgsz,
        min_conf=args.min_conf,
        iou=args.iou,
        device=args.device,
        overwrite=args.overwrite,
    )
    uploaded_dir = _upload_run_dir(bundle_result.output_dir, upload_root) if upload_root else None
    return YoloPredictionExportResult(
        output_dir=bundle_result.output_dir,
        predictions_path=bundle_result.predictions_path,
        total_images=bundle_result.total_images,
        total_boxes=bundle_result.total_boxes,
        uploaded_dir=uploaded_dir,
    )


def summarize_yolo_run(result: YoloRunResult) -> str:
    lines = []
    if result.output_dir:
        lines.append(f"Results: {result.output_dir}")
    if result.model_path:
        lines.append(f"Model: {result.model_path}")
    if result.confusion_matrix_path:
        lines.append(f"Confusion matrix @ conf 0.25: {result.confusion_matrix_path}")
    if result.prediction_bundle_path:
        lines.append(f"Prediction viewer bundle: {result.prediction_bundle_path}")
    if result.uploaded_dir:
        lines.append(f"Uploaded results: {result.uploaded_dir}")
    return "\n".join(lines) if lines else "YOLO run finished."


def summarize_yolo_prediction_export(result: YoloPredictionExportResult) -> str:
    lines = [
        f"Prediction viewer bundle: {result.output_dir}",
        f"Predictions: {result.predictions_path}",
        f"Images: {result.total_images}",
        f"Boxes: {result.total_boxes}",
    ]
    if result.uploaded_dir:
        lines.append(f"Uploaded bundle: {result.uploaded_dir}")
    return "\n".join(lines)


def summarize_yolo_experiment_suite(result: YoloExperimentSuiteResult) -> str:
    lines = [
        f"Experiments: {result.total_experiments}",
        f"Results: {result.output_dir}",
        f"Grid: {result.grid_path}",
        f"Manifest: {result.manifest_path}",
    ]
    if result.uploaded_dir:
        lines.append(f"Uploaded results: {result.uploaded_dir}")
    return "\n".join(lines)


def _training_kwargs(args: Any, data_path: Path, project: Path, name: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "project": str(project),
        "plots": True,
        "save": True,
        "save_period": args.save_period,
        "exist_ok": args.overwrite,
    }
    if name:
        kwargs["name"] = name
    if args.device:
        kwargs["device"] = args.device
    return kwargs


def _base_training_metadata(args: Any) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "save_period": args.save_period,
        "overwrite": args.overwrite,
    }


def _load_yolo(model_name_or_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `conda env update -f environment.yml --prune`."
        ) from exc
    return YOLO(model_name_or_path)


def _check_experiment_folders(
    project: Path,
    experiments: Sequence[YoloExperiment],
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [experiment.name for experiment in experiments if (project / experiment.name).exists()]
    if existing:
        names = ", ".join(existing[:5])
        if len(existing) > 5:
            names += ", ..."
        raise FileExistsError(
            f"Experiment output folders already exist under {project}: {names}. "
            "Pass --overwrite or choose a different --output-dir/--run-prefix."
        )


def _write_tensorboard_scalars(run_dir: Path) -> Path | None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None

    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            epoch = _epoch_from_row(row, row_index)
            for key, value in row.items():
                tag = key.strip()
                if not tag or tag == "epoch":
                    continue
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                writer.add_scalar(tag, scalar, epoch)
    writer.close()
    return tensorboard_dir


def _epoch_from_row(row: dict[str, str], fallback: int) -> int:
    value = ""
    for key, row_value in row.items():
        if key.strip() == "epoch":
            value = row_value
            break
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _write_confidence_confusion_matrix(
    dataset_yaml: Path,
    split: str,
    predictions_json: Path,
    output_dir: Path,
    confidence: float,
    iou_threshold: float = 0.5,
) -> Path:
    if not predictions_json.exists():
        raise FileNotFoundError(
            f"Ultralytics did not write predictions.json, so the conf {confidence:g} "
            f"confusion matrix cannot be created: {predictions_json}"
        )

    config = _read_dataset_yaml(dataset_yaml)
    dataset_root = _dataset_root_from_config(dataset_yaml, config)
    class_names = _class_names_from_config(config)
    image_paths = _split_image_paths(config, dataset_yaml, dataset_root, split)
    image_key_to_stem = _image_key_to_stem(image_paths)
    predictions = _read_prediction_boxes(
        predictions_json,
        image_key_to_stem,
        num_classes=len(class_names),
        confidence=confidence,
    )

    matrix = [[0 for _ in range(len(class_names) + 1)] for _ in range(len(class_names) + 1)]
    for image_path in image_paths:
        true_boxes = _read_ground_truth_boxes(image_path, dataset_root, split)
        pred_boxes = predictions.get(image_path.stem, [])
        _add_image_to_confusion_matrix(matrix, pred_boxes, true_boxes, iou_threshold)

    labels = [*class_names, "background"]
    suffix = _confidence_suffix(confidence)
    csv_path = output_dir / f"confusion_matrix_{suffix}.csv"
    png_path = output_dir / f"confusion_matrix_{suffix}.png"
    _write_confusion_matrix_csv(csv_path, matrix, labels)
    _write_confusion_matrix_png(png_path, matrix, labels, confidence)
    return png_path


def _write_prediction_bundle_for_split(
    model: Any,
    model_path: str,
    dataset_yaml: Path,
    split: str,
    output_dir: Path,
    imgsz: int,
    min_conf: float,
    iou: float,
    device: str | None,
    overwrite: bool,
):
    from ml.yolo_prediction_bundle import write_prediction_bundle

    config = _read_dataset_yaml(dataset_yaml)
    dataset_root = _dataset_root_from_config(dataset_yaml, config)
    class_names = _class_names_from_config(config)
    image_paths = _split_image_paths(config, dataset_yaml, dataset_root, split)
    return write_prediction_bundle(
        model=model,
        image_paths=image_paths,
        output_dir=output_dir,
        model_path=model_path,
        data_path=str(dataset_yaml),
        split=split,
        class_names=class_names,
        imgsz=imgsz,
        min_conf=min_conf,
        iou=iou,
        device=device,
        overwrite=overwrite,
    )


def _read_dataset_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read dataset.yaml. Run `conda env update -f environment.yml --prune`."
        ) from exc
    with path.expanduser().open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return config


def _dataset_root_from_config(dataset_yaml: Path, config: Mapping[str, Any]) -> Path:
    root_value = config.get("path")
    if root_value is None:
        return dataset_yaml.expanduser().resolve().parent
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = dataset_yaml.expanduser().resolve().parent / root
    return root


def _class_names_from_config(config: Mapping[str, Any]) -> list[str]:
    names = config.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        indexed: list[tuple[int, str]] = []
        for key, value in names.items():
            indexed.append((int(key), str(value)))
        indexed.sort(key=lambda item: item[0])
        return [name for _, name in indexed]
    raise ValueError("dataset.yaml must define class names under `names`.")


def _split_image_paths(
    config: Mapping[str, Any],
    dataset_yaml: Path,
    dataset_root: Path,
    split: str,
) -> list[Path]:
    split_value = config.get(split)
    if split_value is None:
        raise ValueError(f"dataset.yaml does not define split {split!r}.")

    paths: list[Path] = []
    values = split_value if isinstance(split_value, list) else [split_value]
    for value in values:
        candidate = _resolve_dataset_path(value, dataset_yaml, dataset_root)
        if candidate.is_dir():
            paths.extend(sorted(p for p in candidate.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        elif candidate.is_file():
            paths.extend(_image_paths_from_text_file(candidate, dataset_root))
        else:
            raise FileNotFoundError(f"Dataset split path not found: {candidate}")
    return sorted(dict.fromkeys(paths), key=lambda p: p.as_posix())


def _resolve_dataset_path(value: Any, dataset_yaml: Path, dataset_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    root_path = dataset_root / path
    if root_path.exists():
        return root_path
    return dataset_yaml.expanduser().resolve().parent / path


def _image_paths_from_text_file(path: Path, dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(image_path)
    return images


def _image_key_to_stem(image_paths: Sequence[Path]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for path in image_paths:
        for key in (path.stem, path.name, path.as_posix(), str(path)):
            keys.setdefault(key, path.stem)
    return keys


def _read_ground_truth_boxes(image_path: Path, dataset_root: Path, split: str) -> list[_EvalBox]:
    label_path = _label_path_for_image(image_path, dataset_root, split)
    if not label_path.exists():
        return []

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to create the confidence-thresholded confusion matrix."
        ) from exc

    with Image.open(image_path) as image:
        width, height = image.size

    boxes: list[_EvalBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height
        boxes.append(_EvalBox(class_id=class_id, xyxy=(x1, y1, x2, y2)))
    return boxes


def _label_path_for_image(image_path: Path, dataset_root: Path, split: str) -> Path:
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        return dataset_root / "labels" / split / f"{image_path.stem}.txt"

    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return (dataset_root.joinpath(*parts)).with_suffix(".txt")
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def _read_prediction_boxes(
    predictions_json: Path,
    image_key_to_stem: Mapping[str, str],
    num_classes: int,
    confidence: float,
) -> dict[str, list[_EvalBox]]:
    data = json.loads(predictions_json.read_text(encoding="utf-8"))
    rows = _prediction_rows(data)
    image_id_to_stem = _image_id_to_stem_from_json(data, image_key_to_stem)
    category_offset = _category_id_offset(rows, num_classes)
    predictions: dict[str, list[_EvalBox]] = {}
    for row in rows:
        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if score < confidence:
            continue

        image_stem = _prediction_image_stem_from_row(row, image_key_to_stem, image_id_to_stem)
        if not image_stem:
            continue

        try:
            class_id = int(row["category_id"]) - category_offset
            x, y, width, height = (float(value) for value in row["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        if class_id < 0 or class_id >= num_classes:
            continue
        predictions.setdefault(image_stem, []).append(
            _EvalBox(class_id=class_id, xyxy=(x, y, x + width, y + height))
        )
    return predictions


def _image_id_to_stem_from_json(data: Any, image_key_to_stem: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(data, Mapping):
        return {}
    images = data.get("images")
    if not isinstance(images, list):
        return {}

    mapping: dict[str, str] = {}
    for image in images:
        if not isinstance(image, Mapping):
            continue
        image_id = image.get("id")
        file_name = (
            image.get("image_name")
            or image.get("file_name")
            or image.get("filename")
            or image.get("image")
            or image.get("path")
        )
        if image_id is None or file_name is None:
            continue
        stem = _prediction_image_stem(str(file_name), image_key_to_stem, {})
        if stem:
            mapping[str(image_id)] = stem
    return mapping


def _prediction_image_stem_from_row(
    row: Mapping[str, Any],
    image_key_to_stem: Mapping[str, str],
    image_id_to_stem: Mapping[str, str],
) -> str | None:
    for key in ("image_name", "file_name", "filename", "image", "path", "image_id"):
        value = row.get(key)
        if value is None:
            continue
        stem = _prediction_image_stem(str(value), image_key_to_stem, image_id_to_stem)
        if stem:
            return stem
    return None


def _prediction_image_stem(
    image_key: str,
    image_key_to_stem: Mapping[str, str],
    image_id_to_stem: Mapping[str, str],
) -> str | None:
    if not image_key:
        return None
    if image_key in image_id_to_stem:
        return image_id_to_stem[image_key]
    if image_key in image_key_to_stem:
        return image_key_to_stem[image_key]
    path = PurePosixPath(image_key)
    if path.name in image_key_to_stem:
        return image_key_to_stem[path.name]
    if path.stem in image_key_to_stem:
        return image_key_to_stem[path.stem]
    if image_key.isdigit():
        return None
    return None


def _prediction_rows(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, Mapping)]
    if isinstance(data, Mapping):
        for key in ("annotations", "predictions", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
    raise ValueError("predictions.json must contain a list of prediction objects.")


def _category_id_offset(rows: Sequence[Mapping[str, Any]], num_classes: int) -> int:
    category_ids: list[int] = []
    for row in rows:
        try:
            category_ids.append(int(row["category_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not category_ids or any(category_id == 0 for category_id in category_ids):
        return 0
    if min(category_ids) >= 1 and max(category_ids) <= num_classes:
        return 1
    return 0


def _add_image_to_confusion_matrix(
    matrix: list[list[int]],
    pred_boxes: Sequence[_EvalBox],
    true_boxes: Sequence[_EvalBox],
    iou_threshold: float,
) -> None:
    background_index = len(matrix) - 1
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_box in enumerate(pred_boxes):
        for true_index, true_box in enumerate(true_boxes):
            iou = _box_iou(pred_box.xyxy, true_box.xyxy)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, true_index))

    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    for _, pred_index, true_index in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or true_index in matched_truth:
            continue
        pred_box = pred_boxes[pred_index]
        true_box = true_boxes[true_index]
        matrix[pred_box.class_id][true_box.class_id] += 1
        matched_predictions.add(pred_index)
        matched_truth.add(true_index)

    for pred_index, pred_box in enumerate(pred_boxes):
        if pred_index not in matched_predictions:
            matrix[pred_box.class_id][background_index] += 1
    for true_index, true_box in enumerate(true_boxes):
        if true_index not in matched_truth:
            matrix[background_index][true_box.class_id] += 1


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _write_confusion_matrix_csv(path: Path, matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted/actual", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_confusion_matrix_png(
    path: Path,
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    confidence: float,
) -> None:
    try:
        import os
        import tempfile

        mpl_config = Path(tempfile.gettempdir()) / "matplotlib"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to create the confidence-thresholded confusion matrix."
        ) from exc

    fig_size = max(6.0, min(16.0, 1.4 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix (conf >= {confidence:g})")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    max_value = max((value for row in matrix for value in row), default=0)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if max_value and value > max_value / 2 else "black"
            ax.text(column_index, row_index, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _confidence_suffix(confidence: float) -> str:
    return f"conf{int(round(confidence * 100)):03d}"


def _write_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    upload_root: str | None,
) -> None:
    _write_json(manifest_path, manifest)
    if upload_root:
        backend = backend_for_locations(upload_root)
        destination = backend.child_uri(backend.normalize_uri(upload_root), manifest_path.name)
        _upload_file(manifest_path, destination)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _upload_file(path: Path, destination_uri: str) -> None:
    backend = backend_for_locations(destination_uri)
    backend.write_bytes(
        destination_uri,
        path.read_bytes(),
        content_type=_content_type(path),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_dataset_yaml(data: str, work_dir: Path) -> Path:
    if not is_gcs_uri(data):
        return Path(data).expanduser()

    dataset_root_uri = _gcs_parent_uri(data)
    local_root = work_dir / "datasets" / _safe_stage_name(dataset_root_uri)
    _download_prefix(dataset_root_uri, local_root)
    dataset_yaml = local_root / PurePosixPath(parse_gcs_uri(data).object_name).name
    _rewrite_dataset_yaml_path(dataset_yaml, local_root)
    return dataset_yaml


def _stage_model(model: str, work_dir: Path) -> str:
    if not is_gcs_uri(model):
        return model

    parsed = parse_gcs_uri(model)
    filename = PurePosixPath(parsed.object_name).name
    local_dir = work_dir / "models" / _safe_stage_name(_gcs_parent_uri(model))
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename
    backend = backend_for_locations(model)
    with backend.open_reader(model) as reader, local_path.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    return str(local_path)


def _download_prefix(uri: StoragePath, local_dir: Path) -> None:
    backend = backend_for_locations(uri)
    normalized_uri = backend.normalize_uri(uri)
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for file in backend.list_files(normalized_uri):
        dest = local_dir / file.relative_path.as_posix()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with backend.open_reader(file.uri) as reader, dest.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _rewrite_dataset_yaml_path(dataset_yaml: Path, dataset_root: Path) -> None:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Staged dataset.yaml not found: {dataset_yaml}")
    lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("path:"):
            lines[index] = f"path: {dataset_root.as_posix()}"
            break
    else:
        lines.insert(0, f"path: {dataset_root.as_posix()}")
    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _local_project_and_upload_root(
    output_dir: str,
    upload_dir: str | None,
    fallback_project: Path,
) -> tuple[Path, str | None]:
    if is_gcs_uri(output_dir):
        return fallback_project, output_dir
    project = Path(output_dir).expanduser()
    return project, upload_dir


def _upload_run_dir(save_dir: Any, upload_root: str | None) -> str | None:
    if not save_dir or not upload_root:
        return None
    local_dir = Path(save_dir)
    backend = backend_for_locations(upload_root)
    upload_root = backend.normalize_uri(upload_root)
    destination_root = backend.child_uri(upload_root, local_dir.name)
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        backend.write_bytes(
            backend.child_uri(destination_root, rel),
            path.read_bytes(),
            content_type=_content_type(path),
        )
    return destination_root


def _result_paths(
    result: Any,
    uploaded_dir: str | None = None,
    confusion_matrix_path: Path | None = None,
    prediction_bundle_path: Path | None = None,
) -> YoloRunResult:
    save_dir = getattr(result, "save_dir", None)
    output_dir = Path(save_dir) if save_dir else None
    best = output_dir / "weights" / "best.pt" if output_dir else None
    last = output_dir / "weights" / "last.pt" if output_dir else None
    model_path = best if best and best.exists() else last if last and last.exists() else None
    return YoloRunResult(
        output_dir=output_dir,
        model_path=model_path,
        uploaded_dir=uploaded_dir,
        confusion_matrix_path=confusion_matrix_path,
        prediction_bundle_path=prediction_bundle_path,
    )


def _gcs_parent_uri(uri: StoragePath) -> str:
    parsed = parse_gcs_uri(uri)
    parent = PurePosixPath(parsed.object_name).parent
    object_name = "" if parent.as_posix() == "." else parent.as_posix()
    return GcsUri(parsed.bucket, object_name).as_uri()


def _safe_stage_name(uri: StoragePath) -> str:
    return str(uri).replace("gs://", "").replace("/", "__").replace(":", "_")


def _content_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".yaml", ".yml"}:
        return "application/x-yaml"
    if suffix == ".txt":
        return "text/plain"
    if suffix == ".json":
        return "application/json"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return None
