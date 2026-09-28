from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ml.yolo_labels import normalize_class_name


NOISE_LABELS = {"noise", "background", "artifact", "artefact"}


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    image_files_seen: int
    label_files_written: int
    boxes_written: int
    boxes_skipped: int


class AnnotationConversionError(ValueError):
    """Raised when review annotations cannot be converted to named YOLO labels."""


def export_named_yolo_from_annotations(
    annotations_path: Path,
    output_dir: Path,
    include_empty_files: bool = True,
) -> ExportResult:
    annotations_path = annotations_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images, annotations = _extract_coco_like_records(data)
    categories = _category_name_by_id(data)
    annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in annotations:
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    boxes_written = 0
    boxes_skipped = 0
    label_files_written = 0

    for image in images:
        image_id = int(image["id"])
        filename = image["file_name"]
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        if width <= 0 or height <= 0:
            raise AnnotationConversionError(f"Image {filename} has invalid dimensions {width}x{height}")

        lines = []
        for annotation in annotations_by_image_id.get(image_id, []):
            status = str(annotation.get("status", "approved")).lower()
            label = _annotation_label(annotation, categories)
            if status != "approved" or _is_noise_label(label):
                boxes_skipped += 1
                continue

            class_name = normalize_class_name(label)
            x_center, y_center, box_width, box_height = _normalize_bbox(
                annotation.get("bbox"), width, height, filename
            )
            lines.append(
                f"{class_name} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
            boxes_written += 1

        if lines or include_empty_files:
            label_path = output_dir / f"{Path(filename).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_files_written += 1

    return ExportResult(
        output_dir=output_dir,
        image_files_seen=len(images),
        label_files_written=label_files_written,
        boxes_written=boxes_written,
        boxes_skipped=boxes_skipped,
    )


def _extract_coco_like_records(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_images = data.get("images")
    if isinstance(raw_images, list):
        images = raw_images
        annotations = data.get("annotations", [])
        return images, annotations

    if isinstance(raw_images, dict):
        images = []
        annotations = []
        ann_id = 1
        for image_id, (filename, image_data) in enumerate(sorted(raw_images.items()), 1):
            images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": image_data.get("width", 0),
                    "height": image_data.get("height", 0),
                }
            )
            for annotation in image_data.get("annotations", []):
                converted = dict(annotation)
                converted.setdefault("id", ann_id)
                converted["image_id"] = image_id
                annotations.append(converted)
                ann_id += 1
        return images, annotations

    raise AnnotationConversionError("Annotations JSON must contain an images list or images object.")


def _category_name_by_id(data: Dict[str, Any]) -> Dict[int, str]:
    categories = {}
    for category in data.get("categories", []):
        if "id" in category and "name" in category:
            categories[int(category["id"])] = str(category["name"])
    return categories


def _annotation_label(annotation: Dict[str, Any], categories: Dict[int, str]) -> str:
    label = str(annotation.get("label") or "").strip()
    if label:
        return label
    category_id = annotation.get("category_id")
    if category_id is not None and int(category_id) in categories:
        return categories[int(category_id)]
    return "parasite"


def _is_noise_label(label: str) -> bool:
    return normalize_class_name(label) in NOISE_LABELS


def _normalize_bbox(
    bbox: Any,
    image_width: float,
    image_height: float,
    filename: str,
) -> Tuple[float, float, float, float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise AnnotationConversionError(f"Annotation for {filename} has invalid bbox: {bbox!r}")

    x, y, width, height = [float(value) for value in bbox]
    if width <= 0 or height <= 0:
        raise AnnotationConversionError(f"Annotation for {filename} has non-positive bbox size: {bbox!r}")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise AnnotationConversionError(f"Annotation for {filename} extends outside image bounds: {bbox!r}")

    return (
        (x + width / 2.0) / image_width,
        (y + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def summarize_export_result(result: ExportResult) -> str:
    return "\n".join(
        [
            f"Output dir:          {result.output_dir}",
            f"Images seen:         {result.image_files_seen}",
            f"Label files written: {result.label_files_written}",
            f"Boxes written:       {result.boxes_written}",
            f"Boxes skipped:       {result.boxes_skipped}",
        ]
    )

