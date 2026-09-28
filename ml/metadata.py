from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageNameMetadata:
    microscope: str
    parasite: str
    image_code: str


def parse_canonical_image_name(filename: str) -> ImageNameMetadata:
    """Parse <microscope>__<parasite>__<code>.<ext> filenames."""
    stem = Path(filename).stem
    parts = stem.split("__")
    if len(parts) != 3:
        raise ValueError(
            "Expected image filename format "
            "<microscope>__<parasite>__<hex_code>.<ext>; "
            f"got {filename!r}"
        )
    microscope, parasite, image_code = parts
    if not all(parts):
        raise ValueError(f"Image filename contains an empty metadata field: {filename!r}")
    return ImageNameMetadata(
        microscope=microscope,
        parasite=parasite,
        image_code=image_code,
    )


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, length: int = 10) -> str:
    payload = stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def path_relative_to(path: Path, root: Path) -> str:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()
