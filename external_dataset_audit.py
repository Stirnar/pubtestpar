from __future__ import annotations

import argparse
import csv
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from external_datasets import (
    IMAGE_SUFFIXES,
    _annotation_files,
    _bbox_from_image_json_object,
    _bbox_from_minmax,
    _coco_categories,
    _coco_images,
    _coerce_bbox,
    _dataset_root_for_annotation,
    _find_image_path,
    _image_json_size,
    _image_name_for_yolo_label,
    _image_name_from_coco,
    _image_size_for_name,
    _is_coco,
    _is_ignored_external_path,
    _is_image_json_annotation,
    _label_from_yolo_filename,
    _read_json,
    _should_use_json_annotation,
    _xml_float,
    _xml_image_path,
    _xml_int,
    _xml_text,
    _yolo_bbox_to_xywh,
    _yolo_class_names_for_label_path,
)


AUDIT_COLUMNS = [
    "dataset",
    "subfolder",
    "annotation_formats",
    "image_files",
    "json_files",
    "xml_files",
    "txt_files",
    "yolo_label_txt_files",
    "csv_files",
    "ignored_files",
    "source_annotation_files",
    "source_annotations",
    "kept_annotations",
    "reviewable_annotations",
    "excluded_annotations",
    "excluded_missing_image_record",
    "excluded_missing_image_file",
    "excluded_bad_bbox",
    "excluded_bad_row",
    "not_reviewable_annotations",
    "not_reviewable_missing_image",
    "not_reviewable_missing_size",
    "missing_referenced_images",
    "missing_class_name_annotations",
    "secondary_annotation_files",
    "secondary_annotations",
    "unsupported_annotation_files",
    "unsupported_annotations",
    "notes",
]


@dataclass
class DatasetAudit:
    dataset: str
    root: Path
    subfolder: str = "."
    counts: Counter[str] = field(default_factory=Counter)
    formats: set[str] = field(default_factory=set)
    missing_images: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)

    def row(self) -> dict[str, str | int]:
        row: dict[str, str | int] = {column: self.counts.get(column, 0) for column in AUDIT_COLUMNS}
        row["dataset"] = self.dataset
        row["subfolder"] = self.subfolder
        row["annotation_formats"] = ";".join(sorted(self.formats))
        row["excluded_annotations"] = self.counts["source_annotations"] - self.counts["kept_annotations"]
        row["not_reviewable_annotations"] = self.counts["kept_annotations"] - self.counts["reviewable_annotations"]
        row["missing_referenced_images"] = len(self.missing_images)
        row["notes"] = "; ".join(sorted(self.notes))
        return row


def external_dataset_audit_csv(
    external_root: Path,
    by_subfolder: bool = False,
    dataset: str | None = None,
) -> str:
    audits = audit_external_datasets(external_root, by_subfolder=by_subfolder, dataset=dataset)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=AUDIT_COLUMNS)
    writer.writeheader()
    for audit in audits:
        writer.writerow(audit.row())
    return output.getvalue()


def audit_external_datasets(
    external_root: Path,
    by_subfolder: bool = False,
    dataset: str | None = None,
) -> list[DatasetAudit]:
    external_root = external_root.expanduser().resolve()
    if dataset:
        dataset_root = (external_root / dataset).resolve()
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise FileNotFoundError(f"External dataset not found: {dataset_root}")
        if by_subfolder:
            return _audit_dataset_by_subfolder(dataset_root, external_root)
        return [_audit_dataset(dataset_root, external_root)]

    audits: list[DatasetAudit] = []
    for dataset_root in sorted(path for path in external_root.iterdir() if path.is_dir()):
        if by_subfolder:
            audits.extend(_audit_dataset_by_subfolder(dataset_root, external_root))
        else:
            audits.append(_audit_dataset(dataset_root, external_root))
    return audits


def _audit_dataset(dataset_root: Path, external_root: Path) -> DatasetAudit:
    audit = DatasetAudit(dataset=dataset_root.name, root=dataset_root)
    _audit_file_inventory(dataset_root, audit)

    xml_paths = _annotation_files(dataset_root, "*.xml")
    primary_files_before_json = audit.counts["source_annotation_files"]
    if xml_paths:
        _audit_xml_files(xml_paths, dataset_root, audit)

    _audit_json_files(dataset_root, external_root, audit, prefer_xml=bool(xml_paths))
    _audit_csv_files(dataset_root, audit)

    if audit.counts["source_annotation_files"] == primary_files_before_json:
        _audit_yolo_files(dataset_root, audit)
    else:
        _audit_secondary_yolo_files(dataset_root, audit)

    if not audit.formats:
        audit.notes.add("no supported primary annotation format found")
    return audit


def _audit_dataset_by_subfolder(dataset_root: Path, external_root: Path) -> list[DatasetAudit]:
    audits: dict[str, DatasetAudit] = {}

    def audit_for(subfolder: str) -> DatasetAudit:
        audit = audits.get(subfolder)
        if audit is None:
            audit = DatasetAudit(dataset=dataset_root.name, root=dataset_root, subfolder=subfolder)
            audits[subfolder] = audit
        return audit

    xml_paths = _annotation_files(dataset_root, "*.xml")
    for xml_path in xml_paths:
        subfolder = _path_parent_subfolder(xml_path, dataset_root)
        _audit_xml_files([xml_path], dataset_root, audit_for(subfolder))

    has_primary_annotations = bool(xml_paths)
    for path in _annotation_files(dataset_root, "*.json"):
        subfolder = _json_annotation_subfolder(path, dataset_root, external_root)
        audit = audit_for(subfolder)
        try:
            data = _read_json(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            audit.counts["unsupported_annotation_files"] += 1
            audit.notes.add("some json files could not be parsed")
            continue

        if _is_generated_xml_conversion(data) and xml_paths:
            _count_secondary_json(path, data, audit)
            continue

        if _is_coco(data):
            if not _should_use_json_annotation(path):
                _count_secondary_json(path, data, audit)
                continue
            has_primary_annotations = True
            _audit_coco_json_by_subfolder(path, dataset_root, audit_for, data)
            continue

        if _is_image_json_annotation(data):
            has_primary_annotations = True
            _audit_image_json(path, dataset_root, audit, data)
            continue

        if _is_via_json(data):
            _count_secondary_json(path, data, audit)
        else:
            audit.counts["unsupported_annotation_files"] += 1

    for labels_path in sorted(dataset_root.rglob("model_label.csv")):
        if _is_ignored_external_path(labels_path):
            continue
        has_primary_annotations = True
        _audit_csv_file(labels_path, dataset_root, audit_for(_path_parent_subfolder(labels_path, dataset_root)))

    for label_path in _annotation_files(dataset_root, "labels/*.txt"):
        subfolder = _yolo_label_subfolder(label_path, dataset_root)
        audit = audit_for(subfolder)
        if has_primary_annotations:
            audit.counts["secondary_annotation_files"] += 1
            audit.counts["secondary_annotations"] += _count_nonempty_lines(label_path)
        else:
            audit.formats.add("yolo")
            audit.counts["source_annotation_files"] += 1
            _audit_yolo_label_file(label_path, dataset_root, audit)

    if not audits:
        audit_for(".").notes.add("no supported primary annotation format found")

    for subfolder, audit in audits.items():
        _audit_file_inventory_for_subfolder(dataset_root, subfolder, audit)
        if (
            not audit.formats
            and audit.counts["source_annotation_files"] == 0
            and audit.counts["secondary_annotation_files"] == 0
            and audit.counts["unsupported_annotation_files"] == 0
        ):
            audit.notes.add("no supported primary annotation format found")

    return [audits[subfolder] for subfolder in sorted(audits)]


def _audit_file_inventory(dataset_root: Path, audit: DatasetAudit) -> None:
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if _is_ignored_external_path(path):
            audit.counts["ignored_files"] += 1
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            audit.counts["image_files"] += 1
        elif suffix == ".json":
            audit.counts["json_files"] += 1
        elif suffix == ".xml":
            audit.counts["xml_files"] += 1
        elif suffix == ".txt":
            audit.counts["txt_files"] += 1
        elif suffix == ".csv":
            audit.counts["csv_files"] += 1

        if suffix == ".txt" and path.parent.name == "labels":
            audit.counts["yolo_label_txt_files"] += 1


def _audit_json_files(dataset_root: Path, external_root: Path, audit: DatasetAudit, prefer_xml: bool) -> None:
    for path in _annotation_files(dataset_root, "*.json"):
        try:
            data = _read_json(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            audit.counts["unsupported_annotation_files"] += 1
            audit.notes.add("some json files could not be parsed")
            continue

        if _is_generated_xml_conversion(data) and prefer_xml:
            _count_secondary_json(path, data, audit)
            continue

        if _is_coco(data):
            if not _should_use_json_annotation(path):
                _count_secondary_json(path, data, audit)
                continue
            dataset_root_for_json = _dataset_root_for_annotation(path, external_root)
            _audit_coco_json(path, dataset_root_for_json, audit, data)
            continue

        if _is_image_json_annotation(data):
            _audit_image_json(path, dataset_root, audit, data)
            continue

        if _is_via_json(data):
            _count_secondary_json(path, data, audit)
        else:
            audit.counts["unsupported_annotation_files"] += 1


def _audit_coco_json(
    path: Path,
    dataset_root: Path,
    audit: DatasetAudit,
    data: Mapping[str, Any],
    annotation_rows: Sequence[Any] | None = None,
) -> None:
    audit.formats.add("coco_json")
    audit.counts["source_annotation_files"] += 1
    categories = _coco_categories(data)
    images = _coco_images(data)

    rows = annotation_rows if annotation_rows is not None else data.get("annotations", [])
    for row in rows:
        audit.counts["source_annotations"] += 1
        if not isinstance(row, Mapping):
            audit.counts["excluded_bad_row"] += 1
            continue

        image = images.get(row.get("image_id"))
        if image is None:
            audit.counts["excluded_missing_image_record"] += 1
            continue

        bbox = _coerce_bbox(row.get("bbox"))
        if not _valid_bbox(bbox):
            audit.counts["excluded_bad_bbox"] += 1
            continue

        image_name = _image_name_from_coco(image)
        local_image = _record_coco_image_match(image_name, path, dataset_root, audit)
        audit.counts["kept_annotations"] += 1
        if local_image and _coco_image_has_size(image, image_name, dataset_root):
            audit.counts["reviewable_annotations"] += 1
        elif local_image:
            audit.counts["not_reviewable_missing_size"] += 1
        else:
            audit.counts["not_reviewable_missing_image"] += 1

        if row.get("category_id") not in categories:
            audit.notes.add("some coco category ids are not present in categories")


def _audit_coco_json_by_subfolder(
    path: Path,
    dataset_root: Path,
    audit_for: Callable[[str], DatasetAudit],
    data: Mapping[str, Any],
) -> None:
    images = _coco_images(data)
    grouped_rows: dict[str, list[Any]] = defaultdict(list)
    fallback = _path_parent_subfolder(path, dataset_root)

    for row in data.get("annotations", []):
        subfolder = fallback
        if isinstance(row, Mapping):
            image = images.get(row.get("image_id"))
            if image is not None:
                subfolder = _coco_image_subfolder(path, dataset_root, image)
        grouped_rows[subfolder].append(row)

    if not grouped_rows:
        grouped_rows[fallback] = []

    for subfolder in sorted(grouped_rows):
        _audit_coco_json(path, dataset_root, audit_for(subfolder), data, grouped_rows[subfolder])


def _audit_image_json(path: Path, dataset_root: Path, audit: DatasetAudit, data: Mapping[str, Any]) -> None:
    audit.formats.add("image_json")
    audit.counts["source_annotation_files"] += 1
    image_name = str(data.get("filename") or path.with_suffix(".jpg").name).replace("\\", "/")
    width, height = _image_json_size(data.get("size"))
    local_image = _record_image_match(image_name, dataset_root, audit)

    for obj in data.get("objects", []):
        audit.counts["source_annotations"] += 1
        if not isinstance(obj, Mapping):
            audit.counts["excluded_bad_row"] += 1
            continue

        bbox = _bbox_from_image_json_object(obj)
        if not _valid_bbox(bbox):
            audit.counts["excluded_bad_bbox"] += 1
            continue

        audit.counts["kept_annotations"] += 1
        if local_image and width > 0 and height > 0:
            audit.counts["reviewable_annotations"] += 1
        elif local_image:
            audit.counts["not_reviewable_missing_size"] += 1
        else:
            audit.counts["not_reviewable_missing_image"] += 1


def _audit_xml_files(xml_paths: list[Path], dataset_root: Path, audit: DatasetAudit) -> None:
    for xml_path in xml_paths:
        audit.formats.add("voc_xml")
        audit.counts["source_annotation_files"] += 1
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            audit.counts["excluded_bad_row"] += 1
            audit.notes.add("some xml files could not be parsed")
            continue

        objects = root.findall("object")
        audit.counts["source_annotations"] += len(objects)
        filename = _xml_text(root, "filename") or xml_path.with_suffix(".jpg").name
        image_path = _xml_image_path(filename, xml_path, dataset_root)
        if image_path is None:
            audit.counts["excluded_missing_image_file"] += len(objects)
            audit.missing_images.add(Path(filename.replace("\\", "/")).name)
            continue

        width = _xml_int(root, "size/width")
        height = _xml_int(root, "size/height")
        if width <= 0 or height <= 0:
            width, height = _image_size_for_name(image_path.name, dataset_root)

        for obj in objects:
            box = obj.find("bndbox")
            if box is None:
                audit.counts["excluded_bad_bbox"] += 1
                continue
            bbox = (
                _xml_float(box, "xmin"),
                _xml_float(box, "ymin"),
                _xml_float(box, "xmax") - _xml_float(box, "xmin"),
                _xml_float(box, "ymax") - _xml_float(box, "ymin"),
            )
            if not _valid_bbox(bbox):
                audit.counts["excluded_bad_bbox"] += 1
                continue
            audit.counts["kept_annotations"] += 1
            if width > 0 and height > 0:
                audit.counts["reviewable_annotations"] += 1
            else:
                audit.counts["not_reviewable_missing_size"] += 1


def _audit_csv_files(dataset_root: Path, audit: DatasetAudit) -> None:
    for labels_path in sorted(dataset_root.rglob("model_label.csv")):
        if _is_ignored_external_path(labels_path):
            continue
        _audit_csv_file(labels_path, dataset_root, audit)


def _audit_csv_file(labels_path: Path, dataset_root: Path, audit: DatasetAudit) -> None:
    classes_path = labels_path.with_name("model_class.csv")
    if not classes_path.exists():
        audit.counts["unsupported_annotation_files"] += 1
        audit.notes.add("some model_label.csv files have no model_class.csv")
        return

    audit.formats.add("csv")
    audit.counts["source_annotation_files"] += 1
    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            audit.counts["source_annotations"] += 1
            image_name = str(row.get("img_name") or "")
            if not image_name:
                audit.counts["excluded_bad_row"] += 1
                continue
            bbox = _bbox_from_minmax(row)
            if not _valid_bbox(bbox):
                audit.counts["excluded_bad_bbox"] += 1
                continue
            local_image = _record_image_match(image_name, dataset_root, audit)
            audit.counts["kept_annotations"] += 1
            if local_image and _image_has_size(image_name, dataset_root):
                audit.counts["reviewable_annotations"] += 1
            elif local_image:
                audit.counts["not_reviewable_missing_size"] += 1
            else:
                audit.counts["not_reviewable_missing_image"] += 1


def _audit_yolo_files(dataset_root: Path, audit: DatasetAudit) -> None:
    for label_path in _annotation_files(dataset_root, "labels/*.txt"):
        audit.formats.add("yolo")
        audit.counts["source_annotation_files"] += 1
        _audit_yolo_label_file(label_path, dataset_root, audit)


def _audit_secondary_yolo_files(dataset_root: Path, audit: DatasetAudit) -> None:
    for label_path in _annotation_files(dataset_root, "labels/*.txt"):
        audit.counts["secondary_annotation_files"] += 1
        audit.counts["secondary_annotations"] += _count_nonempty_lines(label_path)


def _audit_yolo_label_file(label_path: Path, dataset_root: Path, audit: DatasetAudit) -> None:
    image_name = _image_name_for_yolo_label(label_path, dataset_root)
    local_image = _record_image_match(image_name, dataset_root, audit)
    width, height = _image_size_for_name(image_name, dataset_root)
    label_from_filename = _label_from_yolo_filename(label_path)
    class_names = _yolo_class_names_for_label_path(label_path, dataset_root)

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        audit.counts["source_annotations"] += 1
        parts = line.split()
        if len(parts) != 5:
            audit.counts["excluded_bad_row"] += 1
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError:
            audit.counts["excluded_bad_row"] += 1
            continue
        if box_width <= 0 or box_height <= 0:
            audit.counts["excluded_bad_bbox"] += 1
            continue

        if not label_from_filename and class_id not in class_names:
            audit.counts["missing_class_name_annotations"] += 1

        audit.counts["kept_annotations"] += 1
        bbox = _yolo_bbox_to_xywh(x_center, y_center, box_width, box_height, width, height)
        if local_image and width > 0 and height > 0 and _valid_bbox(bbox):
            audit.counts["reviewable_annotations"] += 1
        elif local_image:
            audit.counts["not_reviewable_missing_size"] += 1
        else:
            audit.counts["not_reviewable_missing_image"] += 1


def _record_image_match(image_name: str, dataset_root: Path, audit: DatasetAudit) -> bool:
    if not image_name:
        audit.missing_images.add("<missing image name>")
        return False
    if _find_image_path(image_name, dataset_root):
        return True
    audit.missing_images.add(image_name)
    return False


def _coco_image_has_size(image: Mapping[str, Any], image_name: str, dataset_root: Path) -> bool:
    try:
        width = int(float(image.get("width") or 0))
        height = int(float(image.get("height") or 0))
    except (TypeError, ValueError):
        width = 0
        height = 0
    return (width > 0 and height > 0) or _image_has_size(image_name, dataset_root)


def _image_has_size(image_name: str, dataset_root: Path) -> bool:
    width, height = _image_size_for_name(image_name, dataset_root)
    return width > 0 and height > 0


def _valid_bbox(bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return False
    _, _, width, height = bbox
    return width > 0 and height > 0


def _count_secondary_json(path: Path, data: Any, audit: DatasetAudit) -> None:
    audit.counts["secondary_annotation_files"] += 1
    audit.counts["secondary_annotations"] += _json_annotation_count(data)
    if _is_generated_xml_conversion(data):
        audit.notes.add("generated xml-conversion json counted as secondary")
    elif not _should_use_json_annotation(path):
        audit.notes.add("secondary json skipped because a preferred coco json exists")


def _json_annotation_count(data: Any) -> int:
    if _is_coco(data):
        return _list_len(data.get("annotations"))
    if _is_image_json_annotation(data):
        return _list_len(data.get("objects"))
    if _is_via_json(data):
        return sum(_list_len(row.get("regions")) for row in data.values() if isinstance(row, Mapping))
    return 0


def _is_via_json(data: Any) -> bool:
    return isinstance(data, Mapping) and any(
        isinstance(row, Mapping) and isinstance(row.get("regions"), list)
        for row in data.values()
    )


def _is_generated_xml_conversion(data: Any) -> bool:
    if not isinstance(data, Mapping):
        return False
    info = data.get("info")
    return isinstance(info, Mapping) and info.get("generator") == "external_datasets.py"


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def _audit_file_inventory_for_subfolder(dataset_root: Path, subfolder: str, audit: DatasetAudit) -> None:
    root = dataset_root if subfolder == "." else dataset_root / subfolder
    if subfolder == "." and audit.counts["source_annotation_files"] == 0 and audit.counts["secondary_annotation_files"] > 0:
        _audit_direct_file_inventory(root, audit)
    elif root.exists():
        _audit_file_inventory(root, audit)


def _audit_direct_file_inventory(root: Path, audit: DatasetAudit) -> None:
    if not root.exists():
        return
    for path in root.iterdir():
        if not path.is_file():
            continue
        if _is_ignored_external_path(path):
            audit.counts["ignored_files"] += 1
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            audit.counts["image_files"] += 1
        elif suffix == ".json":
            audit.counts["json_files"] += 1
        elif suffix == ".xml":
            audit.counts["xml_files"] += 1
        elif suffix == ".txt":
            audit.counts["txt_files"] += 1
        elif suffix == ".csv":
            audit.counts["csv_files"] += 1

        if suffix == ".txt" and path.parent.name == "labels":
            audit.counts["yolo_label_txt_files"] += 1


def _json_annotation_subfolder(path: Path, dataset_root: Path, external_root: Path) -> str:
    try:
        annotation_root = _dataset_root_for_annotation(path, external_root)
        if annotation_root != dataset_root:
            return annotation_root.relative_to(dataset_root).as_posix()
    except ValueError:
        pass
    return _path_parent_subfolder(path, dataset_root)


def _coco_image_subfolder(path: Path, dataset_root: Path, image: Mapping[str, Any]) -> str:
    image_name = _image_name_from_coco(image)
    local_image = _find_coco_image_near_annotation(image_name, path, dataset_root)
    if local_image is not None:
        subfolder = _sample_subfolder_for_image_path(local_image, path, dataset_root)
        if subfolder != ".":
            return subfolder

    image_parent = _strip_image_container_leaf(Path(image_name.replace("\\", "/")).parent)
    if image_parent != Path("."):
        return image_parent.as_posix()

    return _path_parent_subfolder(path, dataset_root)


def _find_coco_image_near_annotation(image_name: str, path: Path, dataset_root: Path) -> Path | None:
    value = Path(image_name.replace("\\", "/"))
    candidates: list[Path] = []

    if value.is_absolute():
        candidates.append(dataset_root / value.name)
    else:
        candidates.append(path.parent / value)
        if value.parent == Path("."):
            candidates.append(path.parent / "images" / value.name)
        candidates.append(dataset_root / value)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate

    return None


def _record_coco_image_match(image_name: str, path: Path, dataset_root: Path, audit: DatasetAudit) -> bool:
    if not image_name:
        audit.missing_images.add("<missing image name>")
        return False
    if _find_coco_image_near_annotation(image_name, path, dataset_root):
        return True
    audit.missing_images.add(image_name)
    return False


def _sample_subfolder_for_image_path(image_path: Path, annotation_path: Path, dataset_root: Path) -> str:
    try:
        relative_parent = image_path.parent.relative_to(dataset_root)
    except ValueError:
        return _path_parent_subfolder(annotation_path, dataset_root)

    subfolder = _strip_image_container_leaf(relative_parent)
    if subfolder == Path("."):
        return _path_parent_subfolder(annotation_path, dataset_root)
    return subfolder.as_posix()


def _strip_image_container_leaf(path: Path) -> Path:
    if path.parts and path.parts[-1].lower() in {"image", "images", "imgs", "jpegimages"}:
        return Path(*path.parts[:-1])
    return path


def _yolo_label_subfolder(label_path: Path, dataset_root: Path) -> str:
    try:
        return label_path.parent.parent.relative_to(dataset_root).as_posix()
    except ValueError:
        return "."


def _path_parent_subfolder(path: Path, dataset_root: Path) -> str:
    try:
        relative = path.parent.relative_to(dataset_root).as_posix()
    except ValueError:
        return "."
    return relative or "."


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external parasite dataset annotation usability.")
    parser.add_argument("--external-root", default="./data/external")
    parser.add_argument("--dataset", default=None, help="Optional dataset folder under --external-root to audit.")
    parser.add_argument("--output", default=None, help="Optional CSV output path. Defaults to stdout.")
    parser.add_argument(
        "--by-subfolder",
        action="store_true",
        help="Report one row per dataset subfolder instead of one row per dataset.",
    )
    args = parser.parse_args()

    content = external_dataset_audit_csv(Path(args.external_root), by_subfolder=args.by_subfolder, dataset=args.dataset)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"External dataset audit CSV: {output_path}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
