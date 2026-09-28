from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml.yolo_labels import inspect_yolo_label_file


@dataclass(frozen=True)
class AnnotationPreviewResult:
    output_dir: Path
    summary_path: Path
    previews_written: int


class AnnotationPreviewError(ValueError):
    """Raised when annotation previews cannot be rendered."""


def render_annotation_previews(
    annotations_path: Path,
    labels_dir: Path,
    output_dir: Path,
    images_dir: Optional[Path] = None,
    limit: int = 10,
) -> AnnotationPreviewResult:
    """Draw YOLO labels on copies of source images listed in an annotation JSON."""
    annotations_path = annotations_path.expanduser().resolve()
    labels_dir = labels_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if limit <= 0:
        raise AnnotationPreviewError("limit must be positive.")
    if not labels_dir.exists():
        raise AnnotationPreviewError(f"YOLO label directory not found: {labels_dir}")

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = _extract_images(data)
    resolved_images_dir = images_dir.expanduser().resolve() if images_dir else None

    summary_lines = []
    for image in images[:limit]:
        preview_path, summary = _render_one_preview(
            image=image,
            labels_dir=labels_dir,
            output_dir=output_dir,
            images_dir=resolved_images_dir,
        )
        summary_lines.append(f"{preview_path} | {summary}")

    summary_path = output_dir.parent / f"{output_dir.name}_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return AnnotationPreviewResult(
        output_dir=output_dir,
        summary_path=summary_path,
        previews_written=len(summary_lines),
    )


def summarize_preview_result(result: AnnotationPreviewResult) -> str:
    return "\n".join(
        [
            f"Output dir:       {result.output_dir}",
            f"Summary:          {result.summary_path}",
            f"Previews written: {result.previews_written}",
        ]
    )


def _render_one_preview(
    image: Dict[str, Any],
    labels_dir: Path,
    output_dir: Path,
    images_dir: Optional[Path],
) -> Tuple[Path, str]:
    filename = str(image["file_name"])
    image_path = _resolve_image_path(image, images_dir)
    label_path = labels_dir / f"{Path(filename).stem}.txt"
    if not label_path.exists():
        raise AnnotationPreviewError(f"YOLO label file not found for {filename}: {label_path}")

    inspect_yolo_label_file(label_path)
    boxes = _read_yolo_boxes(label_path)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise AnnotationPreviewError("Pillow is required to render image previews.") from exc

    expected_size = _expected_image_size(image)
    with Image.open(image_path) as source:
        preview = _orient_image_for_annotations(source, expected_size).convert("RGB")

    width, height = preview.size
    draw = ImageDraw.Draw(preview)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    line_width = max(3, round(max(width, height) * 0.002))
    for index, (class_token, x_center, y_center, box_width, box_height) in enumerate(boxes, 1):
        left = (x_center - box_width / 2.0) * width
        top = (y_center - box_height / 2.0) * height
        right = (x_center + box_width / 2.0) * width
        bottom = (y_center + box_height / 2.0) * height
        xy = (
            round(max(0.0, left)),
            round(max(0.0, top)),
            round(min(float(width), right)),
            round(min(float(height), bottom)),
        )
        draw.rectangle(xy, outline=(255, 128, 0), width=line_width)
        text = f"{index}: {class_token}"
        text_position = (xy[0], max(0, xy[1] - 18))
        draw.text(text_position, text, fill=(255, 128, 0), font=font)

    output_path = output_dir / f"{int(image['id']):03d}_{Path(filename).stem}_boxes.png"
    preview.save(output_path)
    summary = (
        f"{image['id']} {filename} boxes={len(boxes)} size={width}x{height} "
        f"image={image_path} labels={label_path}"
    )
    return output_path, summary


def _expected_image_size(image: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _orient_image_for_annotations(source: Any, expected_size: Optional[Tuple[int, int]]) -> Any:
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(source)
    if expected_size is None:
        return image

    expected_width, expected_height = expected_size
    if image.size == expected_size:
        return image

    actual_width, actual_height = image.size
    if (actual_width, actual_height) == (expected_height, expected_width):
        return image.transpose(Image.Transpose.ROTATE_270)

    if _aspect_ratio_matches(actual_width, actual_height, expected_width, expected_height):
        return image
    if _aspect_ratio_matches(actual_width, actual_height, expected_height, expected_width):
        return image.transpose(Image.Transpose.ROTATE_270)

    raise AnnotationPreviewError(
        "Image dimensions do not match annotation dimensions: "
        f"image={actual_width}x{actual_height}, annotations={expected_width}x{expected_height}"
    )


def _aspect_ratio_matches(
    actual_width: int,
    actual_height: int,
    expected_width: int,
    expected_height: int,
    tolerance: float = 0.02,
) -> bool:
    actual_ratio = actual_width / actual_height
    expected_ratio = expected_width / expected_height
    return abs(actual_ratio - expected_ratio) <= tolerance


def _extract_images(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_images = data.get("images")
    if isinstance(raw_images, list):
        return raw_images

    if isinstance(raw_images, dict):
        images = []
        for image_id, (filename, image_data) in enumerate(sorted(raw_images.items()), 1):
            image = dict(image_data)
            image.setdefault("id", image_id)
            image.setdefault("file_name", filename)
            images.append(image)
        return images

    raise AnnotationPreviewError("Annotations JSON must contain an images list or images object.")


def _resolve_image_path(image: Dict[str, Any], images_dir: Optional[Path]) -> Path:
    candidates = []
    for key in ("path", "image_path", "full_path", "flat_path", "raw_path", "file_path"):
        value = image.get(key)
        if value:
            candidates.append(Path(str(value)).expanduser())

    filename = Path(str(image["file_name"]))
    candidates.append(filename.expanduser())

    if images_dir is not None:
        candidates.extend([images_dir / filename, images_dir / filename.name])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise AnnotationPreviewError(
        f"Image file not found for {image['file_name']}. "
        "Pass --images-dir pointing at the image folder. "
        f"Searched: {searched}"
    )


def _read_yolo_boxes(label_path: Path) -> List[Tuple[str, float, float, float, float]]:
    boxes = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            class_token, x_center, y_center, box_width, box_height = line.split()
            boxes.append(
                (
                    class_token,
                    float(x_center),
                    float(y_center),
                    float(box_width),
                    float(box_height),
                )
            )
    return boxes
