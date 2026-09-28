from __future__ import annotations

import csv
import io
import shutil
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from file_io.storage import StorageBackend, StoragePath, backend_for_locations, is_gcs_uri, join_storage_uri, parse_gcs_uri
from ml.metadata import parse_canonical_image_name
from ml.yolo_labels import inspect_yolo_label_bytes, normalize_class_name


DEFAULT_GCS_STORAGE_ROOT = "gs://stool-parasite-images"
SPLITS = ("train", "val", "test")
SELECTION_ORDER = ("val", "test", "train")


@dataclass(frozen=True)
class CandidateImage:
    filename: str
    microscope: str
    parasite: str
    image_uri: str
    label_uri: str
    image_backend: StorageBackend
    label_backend: StorageBackend
    label_bytes: bytes
    num_boxes: int
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedDataset:
    output_dir: Path
    splits_csv_path: Path
    dataset_yaml_path: Path
    counts: dict[str, int]
    upload_dir: str | None = None


class DatasetPrepError(ValueError):
    """Raised when a YOLO dataset cannot be prepared safely."""


def prepare_dataset_from_namespace(args: Any, repo_root: Path) -> PreparedDataset:
    train_images = _required_count(args.train_images, "train_images")
    val_images = _required_count(args.val_images, "val_images")
    test_images = _required_count(args.test_images, "test_images")
    seed = int(args.seed)
    storage_root = _storage_root_from_namespace(args, repo_root)
    default_output = repo_root / "runs" / "datasets" / f"seed{seed}_train{train_images}_val{val_images}_test{test_images}"

    output_dir_arg = args.output_dir
    upload_dir = args.upload_dir
    if output_dir_arg and is_gcs_uri(output_dir_arg):
        upload_dir = output_dir_arg
        output_dir = default_output
    else:
        output_dir = _resolve_local_path(Path(output_dir_arg), repo_root) if output_dir_arg else default_output

    return prepare_dataset(
        repo_root=repo_root,
        master_index_path=_defaulted_storage_path(
            args.master_index,
            storage_root,
            repo_root / "data" / "preprocess" / "metadata" / "master_index.csv",
            ("preprocess", "metadata", "master_index.csv"),
        ),
        images_dir=_defaulted_storage_path(
            args.images_dir,
            storage_root,
            repo_root / "data" / "preprocess" / "flat_images",
            ("preprocess", "flat_images"),
        ),
        labels_dir=_defaulted_storage_path(
            args.labels_dir,
            storage_root,
            repo_root / "data" / "labels" / "canonical",
            ("preprocess", "YOLO_labels", "canonical"),
        ),
        output_dir=output_dir,
        upload_dir=upload_dir,
        class_names=args.class_names or [],
        parasites=args.parasites or [],
        train_microscopes=args.train_microscopes or [],
        val_microscopes=args.val_microscopes or [],
        test_microscopes=args.test_microscopes or [],
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        seed=seed,
        skip_missing_labels=bool(args.skip_missing_labels),
        overwrite=bool(args.overwrite),
    )


def prepare_dataset(
    repo_root: Path,
    master_index_path: StoragePath,
    images_dir: StoragePath,
    labels_dir: StoragePath,
    output_dir: Path,
    class_names: Sequence[str],
    parasites: Sequence[str],
    train_microscopes: Sequence[str],
    val_microscopes: Sequence[str],
    test_microscopes: Sequence[str],
    train_images: int,
    val_images: int,
    test_images: int,
    seed: int,
    skip_missing_labels: bool = False,
    overwrite: bool = False,
    upload_dir: StoragePath | None = None,
) -> PreparedDataset:
    repo_root = repo_root.expanduser().resolve()
    output_dir = _resolve_local_path(output_dir, repo_root)
    class_names = _normalize_class_names(class_names)

    counts = {
        "train": _required_count(train_images, "train_images"),
        "val": _required_count(val_images, "val_images"),
        "test": _required_count(test_images, "test_images"),
    }
    if counts["train"] < 1:
        raise DatasetPrepError("train_images must be at least 1.")

    rows = _read_master_index(master_index_path, repo_root)
    microscope_filters = {
        "train": tuple(train_microscopes),
        "val": tuple(val_microscopes),
        "test": tuple(test_microscopes),
    }
    all_microscopes = _unique([m for values in microscope_filters.values() for m in values])
    candidates = _collect_candidates(
        rows=rows,
        images_dir=images_dir,
        labels_dir=labels_dir,
        repo_root=repo_root,
        parasites=parasites,
        microscopes=all_microscopes,
        skip_missing_labels=skip_missing_labels,
    )
    if not candidates:
        raise DatasetPrepError("No eligible images with valid labels were found.")

    selected = _select_splits(candidates, microscope_filters, counts, seed)
    _validate_selected_labels(selected, class_names)
    _write_dataset(output_dir, selected, class_names, overwrite)
    uploaded_uri = _upload_dataset(output_dir, upload_dir) if upload_dir else None

    flat_selected = [image for split in SPLITS for image in selected[split]]
    result_counts = {
        "train_images": len(selected["train"]),
        "val_images": len(selected["val"]),
        "test_images": len(selected["test"]),
        "total_images": len(flat_selected),
        "total_boxes": sum(image.num_boxes for image in flat_selected),
    }
    return PreparedDataset(
        output_dir=output_dir,
        splits_csv_path=output_dir / "splits.csv",
        dataset_yaml_path=output_dir / "dataset.yaml",
        counts=result_counts,
        upload_dir=uploaded_uri,
    )


def summarize_prepared_dataset(result: PreparedDataset) -> str:
    counts = result.counts
    lines = [
        f"Prepared YOLO dataset: {result.output_dir}",
        f"Images: train={counts['train_images']} val={counts['val_images']} test={counts['test_images']}",
        f"Boxes: {counts['total_boxes']}",
        f"Splits CSV: {result.splits_csv_path}",
        f"Dataset YAML: {result.dataset_yaml_path}",
    ]
    if result.upload_dir:
        lines.append(f"Uploaded dataset: {result.upload_dir}")
    return "\n".join(lines)


def _read_master_index(path: StoragePath, repo_root: Path) -> list[dict[str, str]]:
    backend = _backend_for(path)
    uri = _normalize_storage_uri(path, repo_root, backend)
    if not backend.exists(uri):
        raise DatasetPrepError(f"Master index not found: {uri}")
    content = backend.read_bytes(uri).decode("utf-8")
    return list(csv.DictReader(io.StringIO(content)))


def _collect_candidates(
    rows: Iterable[Mapping[str, str]],
    images_dir: StoragePath,
    labels_dir: StoragePath,
    repo_root: Path,
    parasites: Sequence[str],
    microscopes: Sequence[str],
    skip_missing_labels: bool,
) -> list[CandidateImage]:
    parasite_filter = {_filter_key(value) for value in parasites if value}
    microscope_filter = {_filter_key(value) for value in microscopes if value}
    image_backend = _backend_for(images_dir)
    label_backend = _backend_for(labels_dir)
    images_root = _normalize_storage_uri(images_dir, repo_root, image_backend)
    labels_root = _normalize_storage_uri(labels_dir, repo_root, label_backend)
    candidates: list[CandidateImage] = []
    errors: list[str] = []
    seen_filenames: set[str] = set()

    for row in rows:
        if _is_duplicate_row(row):
            continue

        filename = _filename_from_row(row)
        if not filename:
            continue

        microscope, parasite = _metadata_from_row(row, filename)
        if parasite_filter and _filter_key(parasite) not in parasite_filter:
            continue
        if microscope_filter and _filter_key(microscope) not in microscope_filter:
            continue
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)

        image_uri = image_backend.child_uri(images_root, filename)
        label_uri = label_backend.child_uri(labels_root, f"{Path(filename).stem}.txt")
        if not image_backend.exists(image_uri):
            errors.append(f"Image file not found: {image_uri}")
            continue
        if not label_backend.exists(label_uri):
            if skip_missing_labels:
                continue
            errors.append(f"Label file not found: {label_uri}")
            continue

        try:
            label_bytes = _drop_uncertain_label_lines(label_backend.read_bytes(label_uri))
            stats = inspect_yolo_label_bytes(label_bytes, label_uri)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if stats.num_boxes == 0:
            continue

        candidates.append(
            CandidateImage(
                filename=filename,
                microscope=microscope,
                parasite=parasite,
                image_uri=image_uri,
                label_uri=label_uri,
                image_backend=image_backend,
                label_backend=label_backend,
                label_bytes=label_bytes,
                num_boxes=stats.num_boxes,
                class_ids=stats.class_ids,
                class_names=stats.class_names,
            )
        )

    if errors:
        preview = "\n".join(errors[:10])
        suffix = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more"
        raise DatasetPrepError(f"Dataset validation failed:\n{preview}{suffix}")
    return sorted(candidates, key=lambda image: image.filename)


def _is_duplicate_row(row: Mapping[str, str]) -> bool:
    return row.get("status", "").strip().lower() == "duplicate"


def _drop_uncertain_label_lines(data: bytes) -> bytes:
    kept_lines: list[str] = []
    for raw_line in data.decode("utf-8").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        class_token = parts[0]
        if not _is_uncertain_class_name(class_token):
            kept_lines.append(raw_line.strip())
    return ("\n".join(kept_lines) + "\n").encode("utf-8") if kept_lines else b""


def _is_uncertain_class_name(class_token: str) -> bool:
    try:
        int(class_token)
    except ValueError:
        return normalize_class_name(class_token).startswith("maybe_")
    return False


def _select_splits(
    candidates: Sequence[CandidateImage],
    microscope_filters: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    seed: int,
) -> dict[str, list[CandidateImage]]:
    rng = random.Random(seed)
    selected: dict[str, list[CandidateImage]] = {split: [] for split in SPLITS}
    used: set[str] = set()

    for split in SELECTION_ORDER:
        microscope_filter = {_filter_key(value) for value in microscope_filters.get(split, ()) if value}
        pool = [
            image
            for image in candidates
            if image.filename not in used
            and (not microscope_filter or _filter_key(image.microscope) in microscope_filter)
        ]
        rng.shuffle(pool)
        requested = int(counts[split])
        if len(pool) < requested:
            raise DatasetPrepError(
                f"Requested {requested} {split} images, but only {len(pool)} eligible images are available."
            )
        chosen = sorted(pool[:requested], key=lambda image: image.filename)
        selected[split] = chosen
        used.update(image.filename for image in chosen)

    return selected


def _validate_selected_labels(
    selected: Mapping[str, Sequence[CandidateImage]],
    class_names: Sequence[str],
) -> None:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    errors: list[str] = []
    for images in selected.values():
        for image in images:
            for class_name in image.class_names:
                normalized = normalize_class_name(class_name)
                if normalized not in class_to_id:
                    errors.append(
                        f"{image.label_uri}: class {class_name!r} is not in --class-name."
                    )
            for class_id in image.class_ids:
                if class_id >= len(class_names):
                    errors.append(
                        f"{image.label_uri}: class id {class_id} has no matching --class-name."
                    )
    if errors:
        raise DatasetPrepError("Dataset label classes do not match CLI classes:\n" + "\n".join(errors))


def _write_dataset(
    output_dir: Path,
    selected: Mapping[str, Sequence[CandidateImage]],
    class_names: Sequence[str],
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise DatasetPrepError(f"Output directory already exists and is not empty: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    for split in SPLITS:
        for image in selected[split]:
            _write_image(image, output_dir / "images" / split / image.filename)
            _write_numeric_label(
                image.label_bytes,
                output_dir / "labels" / split / f"{Path(image.filename).stem}.txt",
                class_to_id,
            )

    _write_splits_csv(output_dir / "splits.csv", selected)
    _write_dataset_yaml(output_dir / "dataset.yaml", output_dir, class_names)


def _write_image(image: CandidateImage, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not is_gcs_uri(image.image_uri):
        dest.symlink_to(Path(image.image_uri))
        return
    with image.image_backend.open_reader(image.image_uri) as reader, dest.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _write_numeric_label(data: bytes, dest: Path, class_to_id: Mapping[str, int]) -> None:
    lines: list[str] = []
    for raw_line in data.decode("utf-8").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        class_token, *coords = parts
        try:
            class_id = int(class_token)
        except ValueError:
            class_id = class_to_id[normalize_class_name(class_token)]
        lines.append(" ".join([str(class_id), *coords]))
    dest.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _write_splits_csv(
    path: Path,
    selected: Mapping[str, Sequence[CandidateImage]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split"])
        writer.writeheader()
        for split in SPLITS:
            for image in selected[split]:
                writer.writerow({"filename": image.filename, "split": split})


def _write_dataset_yaml(path: Path, output_dir: Path, class_names: Sequence[str]) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _upload_dataset(local_dir: Path, upload_dir: StoragePath) -> str:
    backend = _backend_for(upload_dir)
    upload_root = backend.normalize_uri(upload_dir)
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        backend.write_bytes(
            backend.child_uri(upload_root, rel),
            path.read_bytes(),
            content_type=_content_type(path),
        )
    return upload_root


def _filename_from_row(row: Mapping[str, str]) -> str:
    for key in ("canonical_name", "flat_path", "filename", "raw_filename"):
        filename = _basename(row.get(key, ""))
        if filename:
            return filename
    return ""


def _basename(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return PurePosixPath(value.replace("\\", "/").rstrip("/")).name


def _metadata_from_row(row: Mapping[str, str], filename: str) -> tuple[str, str]:
    try:
        parsed = parse_canonical_image_name(filename)
    except ValueError:
        return row.get("microscope", ""), row.get("parasite", "")
    return row.get("microscope") or parsed.microscope, row.get("parasite") or parsed.parasite


def _normalize_class_names(values: Sequence[str]) -> tuple[str, ...]:
    class_names = tuple(_unique(normalize_class_name(value) for value in values if value))
    if not class_names:
        raise DatasetPrepError("At least one --class-name is required.")
    return class_names


def _required_count(value: Any, name: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetPrepError(f"{name} must be an integer.") from exc
    if count < 0:
        raise DatasetPrepError(f"{name} must be non-negative.")
    return count


def _defaulted_storage_path(
    value: str | None,
    storage_root: str | None,
    local_default: Path,
    remote_parts: Sequence[str],
) -> StoragePath:
    if value:
        return value
    if storage_root:
        return join_storage_uri(storage_root, *remote_parts)
    return local_default


def _infer_storage_root(*values: str | None) -> str | None:
    for value in values:
        if not value or not is_gcs_uri(value):
            continue
        parsed = parse_gcs_uri(value)
        object_name = parsed.object_name.strip("/")
        for suffix in (
            "preprocess/flat_images",
            "preprocess/metadata/master_index.csv",
            "preprocess/YOLO_labels/canonical",
            "labels/canonical",
        ):
            if object_name == suffix:
                return f"gs://{parsed.bucket}"
            suffix_with_separator = f"/{suffix}"
            if object_name.endswith(suffix_with_separator):
                prefix = object_name[: -len(suffix_with_separator)].strip("/")
                return f"gs://{parsed.bucket}/{prefix}" if prefix else f"gs://{parsed.bucket}"
        return f"gs://{parsed.bucket}"
    return None


def _storage_root_from_namespace(args: Any, repo_root: Path) -> str | None:
    if args.storage_root:
        return args.storage_root

    inferred = _infer_storage_root(args.master_index, args.images_dir, args.labels_dir)
    if inferred:
        return inferred

    if args.master_index or args.images_dir or args.labels_dir:
        return None

    local_master_index = repo_root / "data" / "preprocess" / "metadata" / "master_index.csv"
    if not local_master_index.exists():
        return DEFAULT_GCS_STORAGE_ROOT
    return None


def _normalize_storage_uri(path: StoragePath, repo_root: Path, backend: StorageBackend) -> str:
    if is_gcs_uri(path):
        return backend.normalize_uri(path)
    return str(_resolve_local_path(Path(path), repo_root))


def _resolve_local_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _backend_for(path: StoragePath) -> StorageBackend:
    return backend_for_locations(path)


def _filter_key(value: str) -> str:
    return normalize_class_name(value).lower()


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _content_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".yaml", ".yml"}:
        return "application/x-yaml"
    if suffix == ".txt":
        return "text/plain"
    return None
