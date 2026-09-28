from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ml.yolo_training import _write_confidence_confusion_matrix


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(
    PIL_AVAILABLE and MATPLOTLIB_AVAILABLE,
    "Pillow and matplotlib are required for YOLO eval output tests.",
)
class YoloEvalOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "images" / "test"
        self.labels = self.dataset / "labels" / "test"
        self.output = self.root / "eval"
        self.images.mkdir(parents=True)
        self.labels.mkdir(parents=True)
        self.output.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_confusion_matrix_uses_confidence_threshold(self) -> None:
        from PIL import Image

        for stem, class_id in [
            ("img1", 0),
            ("img2", 1),
            ("img3", 1),
        ]:
            Image.new("RGB", (100, 100), (255, 255, 255)).save(self.images / f"{stem}.jpg")
            (self.labels / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.4 0.4\n",
                encoding="utf-8",
            )

        dataset_yaml = self.dataset / "dataset.yaml"
        dataset_yaml.write_text(
            "\n".join(
                [
                    f"path: {self.dataset.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "names:",
                    "  0: ascaris_lumbricoides",
                    "  1: trichuria_trichuris",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        predictions_json = self.output / "predictions.json"
        predictions_json.write_text(
            json.dumps(
                [
                    {
                        "image_id": "img1",
                        "category_id": 1,
                        "bbox": [30, 30, 40, 40],
                        "score": 0.90,
                    },
                    {
                        "image_id": "img1",
                        "category_id": 2,
                        "bbox": [0, 0, 10, 10],
                        "score": 0.24,
                    },
                    {
                        "image_id": "img2",
                        "category_id": 2,
                        "bbox": [30, 30, 40, 40],
                        "score": 0.80,
                    },
                    {
                        "image_id": "img2",
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "score": 0.70,
                    },
                ]
            ),
            encoding="utf-8",
        )

        png_path = _write_confidence_confusion_matrix(
            dataset_yaml=dataset_yaml,
            split="test",
            predictions_json=predictions_json,
            output_dir=self.output,
            confidence=0.25,
        )

        self.assertTrue(png_path.exists())
        csv_path = self.output / "confusion_matrix_conf025.csv"
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        self.assertEqual(
            rows,
            [
                ["predicted/actual", "ascaris_lumbricoides", "trichuria_trichuris", "background"],
                ["ascaris_lumbricoides", "1", "0", "1"],
                ["trichuria_trichuris", "0", "1", "0"],
                ["background", "0", "1", "0"],
            ],
        )

    def test_confusion_matrix_does_not_guess_numeric_image_ids_by_order(self) -> None:
        from PIL import Image

        for stem in ["a", "b"]:
            Image.new("RGB", (100, 100), (255, 255, 255)).save(self.images / f"{stem}.jpg")
            (self.labels / f"{stem}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

        dataset_yaml = self.dataset / "dataset.yaml"
        dataset_yaml.write_text(
            "\n".join(
                [
                    f"path: {self.dataset.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "names:",
                    "  0: ascaris_lumbricoides",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        predictions_json = self.output / "predictions.json"
        predictions_json.write_text(
            json.dumps([{"image_id": 1, "category_id": 0, "bbox": [30, 30, 40, 40], "score": 0.9}]),
            encoding="utf-8",
        )

        _write_confidence_confusion_matrix(
            dataset_yaml=dataset_yaml,
            split="test",
            predictions_json=predictions_json,
            output_dir=self.output,
            confidence=0.25,
        )

        with (self.output / "confusion_matrix_conf025.csv").open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        self.assertEqual(
            rows,
            [
                ["predicted/actual", "ascaris_lumbricoides", "background"],
                ["ascaris_lumbricoides", "0", "0"],
                ["background", "2", "0"],
            ],
        )

    def test_confusion_matrix_uses_image_name_when_image_id_is_numeric(self) -> None:
        from PIL import Image

        image_path = self.images / "lab_microscope__ascaris_lumbricoides__abc123.jpg"
        Image.new("RGB", (100, 100), (255, 255, 255)).save(image_path)
        (self.labels / f"{image_path.stem}.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n",
            encoding="utf-8",
        )

        dataset_yaml = self.dataset / "dataset.yaml"
        dataset_yaml.write_text(
            "\n".join(
                [
                    f"path: {self.dataset.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "names:",
                    "  0: ascaris_lumbricoides",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        predictions_json = self.output / "predictions.json"
        predictions_json.write_text(
            json.dumps(
                [
                    {
                        "image_id": 1,
                        "image_name": image_path.name,
                        "category_id": 0,
                        "bbox": [30, 30, 40, 40],
                        "score": 0.9,
                    }
                ]
            ),
            encoding="utf-8",
        )

        _write_confidence_confusion_matrix(
            dataset_yaml=dataset_yaml,
            split="test",
            predictions_json=predictions_json,
            output_dir=self.output,
            confidence=0.25,
        )

        with (self.output / "confusion_matrix_conf025.csv").open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        self.assertEqual(
            rows,
            [
                ["predicted/actual", "ascaris_lumbricoides", "background"],
                ["ascaris_lumbricoides", "1", "0"],
                ["background", "0", "0"],
            ],
        )
