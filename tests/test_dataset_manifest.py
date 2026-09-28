from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ml.annotation_previews import render_annotation_previews
from ml.annotations_to_named_yolo import export_named_yolo_from_annotations
from ml.dataset_manifest import (
    DatasetPrepError,
    _defaulted_storage_path,
    _infer_storage_root,
    _storage_root_from_namespace,
    prepare_dataset,
)
from ml.yolo_labels import YoloLabelError, inspect_yolo_label_file


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


class DatasetPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.master_index = self.root / "data" / "preprocess" / "metadata" / "master_index.csv"
        self.images_dir = self.root / "data" / "preprocess" / "flat_images"
        self.labels_dir = self.root / "data" / "labels" / "canonical"
        self.output_root = self.root / "runs" / "datasets"
        self.images_dir.mkdir(parents=True)
        self.labels_dir.mkdir(parents=True)
        self.master_index.parent.mkdir(parents=True)
        self.rows: list[dict[str, str]] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def add_image(
        self,
        index: int,
        microscope: str = "Foldscope",
        parasite: str = "Giardia",
        label: str | None = "giardia_lamblia 0.5 0.5 0.2 0.2\n",
    ) -> str:
        code = f"{index:012x}"
        filename = f"{microscope.lower()}__{parasite.lower()}__{code}.jpg"
        image_path = self.images_dir / filename
        image_path.write_bytes(f"fake image {index}".encode("utf-8"))
        if label is not None:
            (self.labels_dir / f"{Path(filename).stem}.txt").write_text(label, encoding="utf-8")

        self.rows.append(
            {
                "image_id": f"image-{index:04d}",
                "microscope": microscope,
                "parasite": parasite,
                "raw_path": f"data/raw/{filename}",
                "raw_filename": filename,
                "ext": ".jpg",
                "size_bytes": str(image_path.stat().st_size),
                "mtime_epoch": "1",
                "canonical_name": filename,
                "flat_path": f"data/preprocess/flat_images/{filename}",
                "status": "new",
            }
        )
        return filename

    def write_master_index(self) -> None:
        with self.master_index.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    def build(
        self,
        name: str,
        train_images: int,
        val_images: int = 0,
        test_images: int = 0,
        seed: int = 7,
        class_names: list[str] | None = None,
        parasites: list[str] | None = None,
        train_microscopes: list[str] | None = None,
        val_microscopes: list[str] | None = None,
        test_microscopes: list[str] | None = None,
        skip_missing_labels: bool = False,
    ):
        return prepare_dataset(
            repo_root=self.root,
            master_index_path=self.master_index,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            output_dir=self.output_root / name,
            class_names=class_names or ["giardia_lamblia"],
            parasites=parasites or [],
            train_microscopes=train_microscopes or [],
            val_microscopes=val_microscopes or [],
            test_microscopes=test_microscopes or [],
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            seed=seed,
            skip_missing_labels=skip_missing_labels,
        )

    def read_splits(self, result):
        with result.splits_csv_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def split_filenames(self, result, split: str) -> set[str]:
        return {row["filename"] for row in self.read_splits(result) if row["split"] == split}

    def test_prepare_data_writes_splits_and_yolo_tree(self) -> None:
        for idx in range(5):
            self.add_image(idx)
        self.write_master_index()

        result = self.build("basic", train_images=3, val_images=1, test_images=1)

        self.assertEqual(result.counts["train_images"], 3)
        self.assertEqual(result.counts["val_images"], 1)
        self.assertEqual(result.counts["test_images"], 1)
        self.assertEqual(result.counts["total_boxes"], 5)
        self.assertEqual(set(self.read_splits(result)[0].keys()), {"filename", "split"})
        self.assertIn("names:", result.dataset_yaml_path.read_text(encoding="utf-8"))

        train_file = next(iter(self.split_filenames(result, "train")))
        yolo_image = result.output_dir / "images" / "train" / train_file
        yolo_label = result.output_dir / "labels" / "train" / f"{Path(train_file).stem}.txt"
        self.assertTrue(yolo_image.is_symlink())
        self.assertEqual(yolo_label.read_text(encoding="utf-8"), "0 0.5 0.5 0.2 0.2\n")

    def test_master_index_paths_are_used_only_for_filenames(self) -> None:
        filename = self.add_image(0)
        old_local_path = f"/Users/someone/old_repo/data/preprocess/flat_images/{filename}"
        self.rows[-1]["canonical_name"] = old_local_path
        self.rows[-1]["flat_path"] = old_local_path
        self.write_master_index()

        result = self.build("old-paths", train_images=1)

        self.assertEqual(self.split_filenames(result, "train"), {filename})
        self.assertTrue((result.output_dir / "images" / "train" / filename).is_symlink())

    def test_prepare_data_ignores_rows_marked_duplicate(self) -> None:
        keep = self.add_image(0)
        duplicate = self.add_image(1, label=None)
        self.rows[-1]["status"] = "duplicate"
        self.write_master_index()

        result = self.build("ignore-duplicate-status", train_images=1)

        self.assertEqual(self.split_filenames(result, "train"), {keep})
        self.assertNotIn(duplicate, self.split_filenames(result, "train"))

    def test_prepare_data_ignores_repeated_canonical_filenames(self) -> None:
        filename = self.add_image(0)
        self.rows.append({**self.rows[-1], "image_id": "same-canonical-second-row"})
        self.write_master_index()

        result = self.build("ignore-repeated-filename", train_images=1)

        self.assertEqual(self.split_filenames(result, "train"), {filename})

    def test_gcs_image_dir_infers_default_master_index_location(self) -> None:
        storage_root = _infer_storage_root(
            None,
            "gs://stool-parasite-images/preprocess/flat_images",
            None,
        )

        self.assertEqual(storage_root, "gs://stool-parasite-images")
        self.assertEqual(
            _defaulted_storage_path(
                None,
                storage_root,
                self.master_index,
                ("preprocess", "metadata", "master_index.csv"),
            ),
            "gs://stool-parasite-images/preprocess/metadata/master_index.csv",
        )

    def test_missing_local_defaults_fall_back_to_project_bucket(self) -> None:
        self.assertEqual(
            _storage_root_from_namespace(
                SimpleNamespace(
                    storage_root=None,
                    master_index=None,
                    images_dir=None,
                    labels_dir=None,
                ),
                self.root,
            ),
            "gs://stool-parasite-images",
        )
        self.assertEqual(
            _defaulted_storage_path(
                None,
                "gs://stool-parasite-images",
                self.labels_dir,
                ("preprocess", "YOLO_labels", "canonical"),
            ),
            "gs://stool-parasite-images/preprocess/YOLO_labels/canonical",
        )

    def test_prepare_data_can_upload_materialized_dataset(self) -> None:
        for idx in range(2):
            self.add_image(idx)
        self.write_master_index()
        upload_dir = self.root / "uploaded" / "ascaris"

        result = prepare_dataset(
            repo_root=self.root,
            master_index_path=self.master_index,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            output_dir=self.output_root / "uploaded_source",
            upload_dir=upload_dir,
            class_names=["giardia_lamblia"],
            parasites=[],
            train_microscopes=[],
            val_microscopes=[],
            test_microscopes=[],
            train_images=2,
            val_images=0,
            test_images=0,
            seed=7,
        )

        train_file = next(iter(self.split_filenames(result, "train")))
        self.assertTrue((upload_dir / "dataset.yaml").exists())
        self.assertTrue((upload_dir / "splits.csv").exists())
        self.assertEqual(
            (upload_dir / "labels" / "train" / f"{Path(train_file).stem}.txt").read_text(encoding="utf-8"),
            "0 0.5 0.5 0.2 0.2\n",
        )
        self.assertFalse((upload_dir / "images" / "train" / train_file).is_symlink())

    def test_eval_splits_stay_fixed_when_train_count_grows(self) -> None:
        for idx in range(12):
            self.add_image(idx, microscope="Lab_Microscope")
        self.write_master_index()

        small = self.build("small", train_images=3, val_images=2, test_images=2, seed=31)
        large = self.build("large", train_images=6, val_images=2, test_images=2, seed=31)

        val = self.split_filenames(small, "val")
        test = self.split_filenames(small, "test")
        self.assertEqual(val, self.split_filenames(large, "val"))
        self.assertEqual(test, self.split_filenames(large, "test"))
        self.assertTrue(self.split_filenames(small, "train").issubset(self.split_filenames(large, "train")))
        self.assertTrue(self.split_filenames(large, "train").isdisjoint(val | test))
        self.assertTrue(val.isdisjoint(test))

    def test_microscope_filters_hold_out_test_images(self) -> None:
        for idx in range(3):
            self.add_image(idx, microscope="Foldscope")
        for idx in range(3, 5):
            self.add_image(idx, microscope="Lab_Microscope")
        self.write_master_index()

        result = self.build(
            "heldout",
            train_images=2,
            val_images=1,
            test_images=2,
            train_microscopes=["Foldscope"],
            val_microscopes=["Foldscope"],
            test_microscopes=["Lab_Microscope"],
        )

        train_files = self.split_filenames(result, "train")
        val_files = self.split_filenames(result, "val")
        test_files = self.split_filenames(result, "test")
        self.assertTrue(all(name.startswith("foldscope__") for name in train_files | val_files))
        self.assertTrue(all(name.startswith("lab_microscope__") for name in test_files))

    def test_multiclass_named_labels_are_written_as_numeric_ids(self) -> None:
        self.add_image(
            0,
            parasite="Ascaris_Lumbricoides",
            label="ascaris_lumbricoides 0.5 0.5 0.2 0.2\n",
        )
        self.add_image(
            1,
            parasite="Giardia_Lamblia",
            label="giardia_lamblia 0.5 0.5 0.2 0.2\n",
        )
        self.write_master_index()

        result = self.build(
            "multi",
            train_images=2,
            class_names=["ascaris_lumbricoides", "giardia_lamblia"],
        )

        labels = sorted((result.output_dir / "labels" / "train").glob("*.txt"))
        self.assertEqual(labels[0].read_text(encoding="utf-8"), "0 0.5 0.5 0.2 0.2\n")
        self.assertEqual(labels[1].read_text(encoding="utf-8"), "1 0.5 0.5 0.2 0.2\n")

    def test_prepare_data_drops_maybe_label_lines(self) -> None:
        self.add_image(
            0,
            parasite="Trichuria_Trichuris",
            label=(
                "trichuria_trichuris 0.5 0.5 0.2 0.2\n"
                "maybe_trichuria_trichuris 0.4 0.4 0.1 0.1\n"
            ),
        )
        self.write_master_index()

        result = self.build(
            "drop-maybe",
            train_images=1,
            class_names=["trichuria_trichuris"],
            parasites=["trichuria_trichuris"],
        )

        label_path = next((result.output_dir / "labels" / "train").glob("*.txt"))
        self.assertEqual(label_path.read_text(encoding="utf-8"), "0 0.5 0.5 0.2 0.2\n")
        self.assertEqual(result.counts["total_boxes"], 1)

    def test_prepare_data_skips_images_with_only_maybe_labels(self) -> None:
        keep = self.add_image(
            0,
            parasite="Trichuria_Trichuris",
            label="trichuria_trichuris 0.5 0.5 0.2 0.2\n",
        )
        skipped = self.add_image(
            1,
            parasite="Trichuria_Trichuris",
            label="maybe_trichuria_trichuris 0.4 0.4 0.1 0.1\n",
        )
        self.write_master_index()

        result = self.build(
            "skip-only-maybe",
            train_images=1,
            class_names=["trichuria_trichuris"],
            parasites=["trichuria_trichuris"],
        )

        self.assertEqual(self.split_filenames(result, "train"), {keep})
        self.assertNotIn(skipped, self.split_filenames(result, "train"))

    def test_filters_accept_underscore_names_for_space_separated_index_values(self) -> None:
        self.add_image(0, microscope="Lab_Microscope", parasite="Ascaris_Lumbricoides")
        self.rows[-1]["microscope"] = "Lab Microscope"
        self.rows[-1]["parasite"] = "Ascaris Lumbricoides"
        self.write_master_index()

        result = self.build(
            "spaces",
            train_images=1,
            parasites=["Ascaris_Lumbricoides"],
            train_microscopes=["Lab_Microscope"],
        )

        self.assertEqual(result.counts["train_images"], 1)

    def test_missing_label_fails_unless_explicitly_skipped(self) -> None:
        self.add_image(0)
        self.add_image(1, label=None)
        self.write_master_index()

        with self.assertRaisesRegex(DatasetPrepError, "Label file not found"):
            self.build("missing", train_images=1)

        result = self.build("skipped", train_images=1, skip_missing_labels=True)
        self.assertEqual(result.counts["train_images"], 1)

    def test_invalid_yolo_label_is_rejected(self) -> None:
        label_path = self.labels_dir / "bad.txt"
        label_path.write_text("giardia_lamblia 0.95 0.95 0.2 0.2\n", encoding="utf-8")

        with self.assertRaisesRegex(YoloLabelError, "outside normalized"):
            inspect_yolo_label_file(label_path)

    def test_named_yolo_label_is_validated(self) -> None:
        label_path = self.labels_dir / "named.txt"
        label_path.write_text("giardia_lamblia 0.500000 0.350000 0.200000 0.100000\n", encoding="utf-8")

        stats = inspect_yolo_label_file(label_path)

        self.assertEqual(stats.num_boxes, 1)
        self.assertEqual(stats.class_ids, ())
        self.assertEqual(stats.class_names, ("giardia_lamblia",))

    def test_export_review_annotations_to_named_yolo_files(self) -> None:
        annotations_path = self.root / "annotations.json"
        annotations_path.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "Foldscope__Giardia__abc123.jpg", "width": 100, "height": 100},
                        {"id": 2, "file_name": "Foldscope__Giardia__def456.jpg", "width": 100, "height": 100},
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "bbox": [40, 30, 20, 10],
                            "status": "approved",
                            "label": "Giardia Lamblia",
                        },
                        {
                            "id": 2,
                            "image_id": 1,
                            "bbox": [10, 10, 5, 5],
                            "status": "rejected",
                            "label": "Giardia Lamblia",
                        },
                        {
                            "id": 3,
                            "image_id": 1,
                            "bbox": [20, 20, 5, 5],
                            "status": "approved",
                            "label": "Noise",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        output_dir = self.root / "data" / "labels" / "canonical"

        result = export_named_yolo_from_annotations(annotations_path, output_dir)

        self.assertEqual(result.image_files_seen, 2)
        self.assertEqual(result.label_files_written, 2)
        self.assertEqual(result.boxes_written, 1)
        self.assertEqual(result.boxes_skipped, 2)
        self.assertEqual(
            (output_dir / "Foldscope__Giardia__abc123.txt").read_text(encoding="utf-8"),
            "giardia_lamblia 0.500000 0.350000 0.200000 0.100000\n",
        )
        self.assertEqual((output_dir / "Foldscope__Giardia__def456.txt").read_text(encoding="utf-8"), "")

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for annotation preview tests.")
    def test_render_annotation_previews_draws_yolo_boxes_on_image_copies(self) -> None:
        from PIL import Image

        Image.new("RGB", (100, 100), (255, 255, 255)).save(self.images_dir / "one.jpg")
        Image.new("RGB", (100, 100), (255, 255, 255)).save(self.images_dir / "two.jpg")
        (self.labels_dir / "one.txt").write_text(
            "giardia_lamblia 0.500000 0.350000 0.200000 0.100000\n",
            encoding="utf-8",
        )
        (self.labels_dir / "two.txt").write_text("", encoding="utf-8")
        annotations_path = self.root / "annotations.json"
        annotations_path.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100},
                        {"id": 2, "file_name": "two.jpg", "width": 100, "height": 100},
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_dir = self.root / "tests" / "sanity_output" / "previews"

        result = render_annotation_previews(
            annotations_path=annotations_path,
            labels_dir=self.labels_dir,
            output_dir=output_dir,
            images_dir=self.images_dir,
            limit=2,
        )

        self.assertEqual(result.previews_written, 2)
        self.assertTrue((output_dir / "001_one_boxes.png").exists())
        self.assertIn("boxes=1", result.summary_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for annotation preview tests.")
    def test_render_annotation_previews_uses_image_path_from_json(self) -> None:
        from PIL import Image

        image_path = self.images_dir / "one.jpg"
        Image.new("RGB", (100, 100), (255, 255, 255)).save(image_path)
        (self.labels_dir / "one.txt").write_text(
            "giardia_lamblia 0.500000 0.350000 0.200000 0.100000\n",
            encoding="utf-8",
        )
        annotations_path = self.root / "annotations.json"
        annotations_path.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "one.jpg",
                            "image_path": str(image_path),
                            "width": 100,
                            "height": 100,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_dir = self.root / "tests" / "sanity_output" / "previews"

        result = render_annotation_previews(
            annotations_path=annotations_path,
            labels_dir=self.labels_dir,
            output_dir=output_dir,
            limit=1,
        )

        self.assertEqual(result.previews_written, 1)
        self.assertTrue((output_dir / "001_one_boxes.png").exists())


if __name__ == "__main__":
    unittest.main()
