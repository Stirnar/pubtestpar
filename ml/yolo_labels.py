from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


class YoloLabelError(ValueError):
    """Raised when a YOLO label file is malformed."""


@dataclass(frozen=True)
class YoloLabelStats:
    label_path: Path
    checksum: str
    num_boxes: int
    class_ids: Tuple[int, ...]
    class_names: Tuple[str, ...] = ()


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inspect_yolo_label_file(path: Path) -> YoloLabelStats:
    """Validate a numeric or named YOLO label file and return compact stats."""
    return inspect_yolo_label_text(path.read_text(encoding="utf-8"), str(path), checksum_file(path))


def inspect_yolo_label_bytes(data: bytes, label_name: str) -> YoloLabelStats:
    checksum = hashlib.sha256(data).hexdigest()
    return inspect_yolo_label_text(data.decode("utf-8"), label_name, checksum)


def inspect_yolo_label_text(text: str, label_name: str, checksum: str = "") -> YoloLabelStats:
    """Validate numeric or named YOLO label text and return compact stats."""
    class_ids = []
    class_names = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise YoloLabelError(
                f"{label_name}: line {line_number} has {len(parts)} fields; expected 5"
            )

        class_token, *coord_tokens = parts
        try:
            class_id = int(class_token)
        except ValueError as exc:
            if not is_valid_class_name(class_token):
                raise YoloLabelError(
                    f"{label_name}: line {line_number} class token must be a non-negative "
                    "integer or a name like trichuria_trichuris"
                ) from exc
            class_id = None
            class_names.append(class_token)
        else:
            if class_id < 0:
                raise YoloLabelError(
                    f"{label_name}: line {line_number} class id must be non-negative"
                )
            class_ids.append(class_id)

        try:
            x_center, y_center, width, height = [float(token) for token in coord_tokens]
        except ValueError as exc:
            raise YoloLabelError(
                f"{label_name}: line {line_number} coordinates must be numeric"
            ) from exc

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise YoloLabelError(
                f"{label_name}: line {line_number} center coordinates must be in [0, 1]"
            )
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise YoloLabelError(
                f"{label_name}: line {line_number} box width/height must be in (0, 1]"
            )

        tolerance = 1e-6
        if x_center - width / 2 < -tolerance or x_center + width / 2 > 1 + tolerance:
            raise YoloLabelError(
                f"{label_name}: line {line_number} box extends outside normalized x bounds"
            )
        if y_center - height / 2 < -tolerance or y_center + height / 2 > 1 + tolerance:
            raise YoloLabelError(
                f"{label_name}: line {line_number} box extends outside normalized y bounds"
            )

    return YoloLabelStats(
        label_path=Path(label_name),
        checksum=checksum,
        num_boxes=len(class_ids) + len(class_names),
        class_ids=tuple(sorted(set(class_ids))),
        class_names=tuple(sorted(set(class_names))),
    )


def normalize_class_name(label: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower()).strip("_")
    return token or "parasite"


def is_valid_class_name(token: str) -> bool:
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", token) is not None
