from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from file_io.flatten_and_index import (
    _resolve_db_locations,
    canonical_filename,
    copy_if_needed,
    flatten_and_index,
    sha256_file,
)
from file_io.storage import join_storage_uri, parse_gcs_uri

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


class FlattenAndIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for EXIF orientation tests.")
    def test_copy_if_needed_preserves_exif_oriented_image_bytes(self) -> None:
        from PIL import Image

        src = self.root / "source.jpg"
        dst = self.root / "flat" / "source.jpg"
        image = Image.new("RGB", (100, 50), (255, 255, 255))
        exif = image.getexif()
        exif[274] = 6
        image.save(src, exif=exif)

        status = copy_if_needed(src, dst)

        self.assertEqual(status, "new")
        self.assertEqual(dst.read_bytes(), src.read_bytes())
        self.assertEqual(copy_if_needed(src, dst), "copied_exists")

    def test_copy_if_needed_replaces_existing_file_when_bytes_differ(self) -> None:
        src = self.root / "source.jpg"
        dst = self.root / "flat" / "source.jpg"
        src.write_bytes(b"source image bytes")
        dst.parent.mkdir(parents=True)
        dst.write_bytes(b"old rewritten bytes")

        status = copy_if_needed(src, dst)

        self.assertEqual(status, "updated")
        self.assertEqual(dst.read_bytes(), src.read_bytes())

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for EXIF orientation tests.")
    def test_flatten_preserves_exif_oriented_image_bytes(self) -> None:
        from PIL import Image

        raw_root = self.root / "raw"
        flat_out = self.root / "flat"
        csv_out = self.root / "metadata" / "master_index.csv"
        src = raw_root / "Lab Microscope" / "Trichuria Trichuris" / "sample.jpg"
        src.parent.mkdir(parents=True)
        image = Image.new("RGB", (100, 50), (255, 255, 255))
        exif = image.getexif()
        exif[274] = 6
        image.save(src, exif=exif)

        results = flatten_and_index(
            raw_root=raw_root,
            flat_out=flat_out,
            csv_out=csv_out,
            rehash_all=False,
            keep_duplicates=False,
        )

        self.assertEqual([result.status for result in results], ["new"])
        dst = Path(results[0].flat_path)
        self.assertEqual(dst.read_bytes(), src.read_bytes())

    def test_rehash_all_does_not_mark_existing_row_as_its_own_duplicate(self) -> None:
        raw_root = self.root / "raw"
        flat_out = self.root / "flat"
        csv_out = self.root / "metadata" / "master_index.csv"
        src = raw_root / "Lab Microscope" / "Trichuria Trichuris" / "sample.jpg"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"indexable image bytes")

        image_id = sha256_file(src)
        canonical = canonical_filename("Lab Microscope", "Trichuria Trichuris", image_id, ".jpg")
        dst = flat_out / canonical
        dst.parent.mkdir(parents=True)
        dst.write_bytes(src.read_bytes())
        self._write_master_row(csv_out, src, image_id, canonical, dst)

        results = flatten_and_index(
            raw_root=raw_root,
            flat_out=flat_out,
            csv_out=csv_out,
            rehash_all=True,
            keep_duplicates=False,
        )

        self.assertEqual([result.status for result in results], ["updated"])

    def test_flatten_indexes_new_layout_with_three_part_canonical_names(self) -> None:
        raw_root = self.root / "raw"
        flat_out = self.root / "flat"
        csv_out = self.root / "metadata" / "master_index.csv"
        src = raw_root / "Lab Microscope" / "Trichuria Trichuris" / "nested" / "sample.jpg"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"indexable image bytes")

        results = flatten_and_index(
            raw_root=raw_root,
            flat_out=flat_out,
            csv_out=csv_out,
            rehash_all=False,
            keep_duplicates=False,
        )

        image_id = sha256_file(src)
        canonical = canonical_filename("Lab Microscope", "Trichuria Trichuris", image_id, ".jpg")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].microscope, "Lab Microscope")
        self.assertEqual(results[0].parasite, "Trichuria Trichuris")
        self.assertEqual(results[0].canonical_name, canonical)
        self.assertTrue((flat_out / canonical).exists())

        with csv_out.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(
                reader.fieldnames,
                [
                    "image_id",
                    "microscope",
                    "parasite",
                    "raw_path",
                    "raw_filename",
                    "ext",
                    "size_bytes",
                    "mtime_epoch",
                    "canonical_name",
                    "flat_path",
                    "status",
                ],
            )
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["microscope"], "Lab Microscope")
        self.assertEqual(rows[0]["parasite"], "Trichuria Trichuris")
        self.assertEqual(rows[0]["canonical_name"], canonical)

    def test_gcs_storage_root_builds_db_locations(self) -> None:
        raw_root, flat_out, csv_out = _resolve_db_locations(
            repo_root=None,
            storage_root="gs://stool-parasite-images/",
            raw_root=None,
            flat_out=None,
            csv_out=None,
        )

        self.assertEqual(raw_root, "gs://stool-parasite-images/raw")
        self.assertEqual(flat_out, "gs://stool-parasite-images/preprocess/flat_images")
        self.assertEqual(csv_out, "gs://stool-parasite-images/preprocess/metadata/master_index.csv")

    def test_gcs_uri_helpers_preserve_spaces_in_object_names(self) -> None:
        uri = join_storage_uri("gs://stool-parasite-images/raw", "Lab Microscope", "Giardia", "sample 1.jpg")
        parsed = parse_gcs_uri(uri)

        self.assertEqual(uri, "gs://stool-parasite-images/raw/Lab Microscope/Giardia/sample 1.jpg")
        self.assertEqual(parsed.bucket, "stool-parasite-images")
        self.assertEqual(parsed.object_name, "raw/Lab Microscope/Giardia/sample 1.jpg")

    def _write_master_row(self, csv_out: Path, src: Path, image_id: str, canonical: str, dst: Path) -> None:
        csv_out.parent.mkdir(parents=True)
        fieldnames = [
            "image_id",
            "microscope",
            "parasite",
            "raw_path",
            "raw_filename",
            "ext",
            "size_bytes",
            "mtime_epoch",
            "canonical_name",
            "flat_path",
            "status",
        ]
        row = {
            "image_id": image_id,
            "microscope": "Lab Microscope",
            "parasite": "Trichuria Trichuris",
            "raw_path": str(src.resolve()),
            "raw_filename": src.name,
            "ext": src.suffix,
            "size_bytes": str(src.stat().st_size),
            "mtime_epoch": str(int(src.stat().st_mtime)),
            "canonical_name": canonical,
            "flat_path": str(dst.resolve()),
            "status": "new",
        }
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
