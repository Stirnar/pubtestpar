from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


MASTER_INDEX = Path("data/preprocess/metadata/master_index.csv")
RAW_ROOT = Path("data/raw")
FLAT_ROOT = Path("data/preprocess/flat_images")
REPORT = Path("runs/verification/flat_image_identity.csv")


def images_are_identical(image1: Path, image2: Path) -> bool:
    img1 = np.array(Image.open(image1).convert("RGB"))
    img2 = np.array(Image.open(image2).convert("RGB"))

    return img1.shape == img2.shape and np.array_equal(img1, img2)


def raw_path_from_row(row: dict[str, str]) -> Path:
    parts = Path(row["raw_path"]).parts
    if "raw" in parts:
        raw_index = parts.index("raw")
        return RAW_ROOT.joinpath(*parts[raw_index + 1 :])
    return RAW_ROOT / row["microscope"] / row["parasite"] / row["raw_filename"]


def flat_path_from_row(row: dict[str, str]) -> Path:
    return FLAT_ROOT / row["canonical_name"]


def verify_all_images() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with MASTER_INDEX.open(newline="", encoding="utf-8") as index_file, REPORT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as report_file:
        reader = csv.DictReader(index_file)
        writer = csv.DictWriter(
            report_file,
            fieldnames=["raw_path", "flat_path", "same", "error"],
        )
        writer.writeheader()

        total = 0
        same_count = 0

        for row in reader:
            total += 1
            raw_path = raw_path_from_row(row)
            flat_path = flat_path_from_row(row)

            try:
                same = images_are_identical(raw_path, flat_path)
                error = ""
            except Exception as exc:
                same = False
                error = str(exc)

            if same:
                same_count += 1

            writer.writerow(
                {
                    "raw_path": raw_path,
                    "flat_path": flat_path,
                    "same": same,
                    "error": error,
                }
            )

    print(f"Checked {total} images")
    print(f"Identical: {same_count}")
    print(f"Different or errors: {total - same_count}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    verify_all_images()
