#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from file_io.storage import (
    StorageBackend,
    StorageFile,
    StoragePath,
    backend_for_locations,
    is_gcs_uri,
    join_storage_uri,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_GCS_STORAGE_ROOT = "gs://stool-parasite-images"
STORAGE_ROOT_ENV = "PARASITES_STORAGE_ROOT"


@dataclass(frozen=True)
class Record:
    image_id: str
    microscope: str
    parasite: str
    raw_path: str
    raw_filename: str
    ext: str
    size_bytes: int
    mtime_epoch: int
    canonical_name: str
    flat_path: str
    # "new" | "updated" | "duplicate" | "copied_exists" | "skipped_unchanged"
    status: str


def find_repo_root(start: Path) -> Path:
    """
    Find the repository root by walking upward until we find 'data/raw'.
    This avoids needing CLI paths.
    """
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "data" / "raw").exists():
            return candidate
    raise RuntimeError(
        "Could not locate repo root containing 'data/raw'. "
        "Run this script from somewhere inside the repo, or pass --repo-root."
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute the SHA256 hash of a file in a memory-efficient manner.

    This function reads the file in binary mode and processes it in fixed-size
    chunks (default: 1 MB) to avoid loading the entire file into memory.
    The resulting SHA256 digest serves as a deterministic, content-based
    identifier for the file.

    The hash value is used as a stable image ID in the dataset indexing
    pipeline. Two files with identical byte content will produce the same
    hash, while even a single-byte difference will yield a completely
    different hash. This ensures reliable duplicate detection and
    reproducible dataset tracking independent of filenames.

    Parameters
    ----------
    path : Path
        Path to the file to be hashed.
    chunk_size : int, optional
        Number of bytes to read at a time (default is 1 MB).
        Larger values may improve speed for very large files,
        while smaller values reduce peak memory usage.

    Returns
    -------
    str
        Hexadecimal SHA256 digest of the file contents.

    Notes
    -----
    - The file is opened in binary mode ("rb") to ensure exact byte-level hashing.
    - This method is suitable for large microscopy images (e.g., TIFF files).
    - SHA256 is chosen for its extremely low collision probability and
      deterministic behavior.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_storage_file(backend: StorageBackend, file: StorageFile, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with backend.open_reader(file.uri) as reader:
        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def parse_image_metadata(raw_root: StoragePath, file_path: StoragePath | StorageFile) -> Tuple[str, str]:
    """
    Extract (microscope, parasite) from:
      raw_root/microscope/parasite/...
    """
    if isinstance(file_path, StorageFile):
        parts = file_path.relative_path.parts
    elif is_gcs_uri(raw_root) or is_gcs_uri(file_path):
        raise TypeError("GCS metadata parsing requires a StorageFile from scan_raw_images().")
    else:
        rel = Path(file_path).relative_to(Path(raw_root))
        parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"Path does not match <microscope>/<parasite>/...: {file_path}")
    microscope, parasite = parts[0], parts[1]
    return microscope, parasite


def safe_token(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s.strip())
    return out or "unknown"


def canonical_filename(microscope: str, parasite: str, image_id: str, ext: str) -> str:
    # short hash keeps names readable
    return f"{safe_token(microscope).lower()}__{safe_token(parasite).lower()}__{image_id[:12]}{ext.lower()}"


def read_existing_csv(
    csv_path: StoragePath,
    backend: Optional[StorageBackend] = None,
) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    """
    Returns:
      - by_raw_path: raw_path -> row dict
      - known_hashes: set(image_id)
    """
    backend = backend or backend_for_locations(csv_path)
    csv_uri = backend.normalize_uri(csv_path)
    if not backend.exists(csv_uri):
        return {}, set()

    by_raw_path: Dict[str, Dict[str, str]] = {}
    known_hashes: Set[str] = set()

    content = backend.read_bytes(csv_uri).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        rp = row.get("raw_path", "")
        if rp:
            by_raw_path[rp] = row
        iid = row.get("image_id", "")
        if iid:
            known_hashes.add(iid)
    return by_raw_path, known_hashes


def write_csv(
    csv_path: StoragePath,
    rows_by_raw: Dict[str, Dict[str, str]],
    backend: Optional[StorageBackend] = None,
) -> None:
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

    rows = list(rows_by_raw.values())
    rows.sort(key=lambda r: (r.get("microscope", ""), r.get("parasite", ""), r.get("raw_path", "")))

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

    backend = backend or backend_for_locations(csv_path)
    backend.write_bytes(
        backend.normalize_uri(csv_path),
        out.getvalue().encode("utf-8"),
        content_type="text/csv",
    )


def copy_if_needed(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if _files_equal(src, dst):
            return "copied_exists"
        _copy_file_bytes(src, dst)
        return "updated"
    _copy_file_bytes(src, dst)
    return "new"


def _copy_file_bytes(src: Path, dst: Path) -> None:
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        return _streams_equal(left_file, right_file)


def copy_storage_if_needed(backend: StorageBackend, src: StorageFile, dst_uri: str) -> str:
    if backend.exists(dst_uri):
        if _storage_objects_equal(backend, src.uri, dst_uri):
            return "copied_exists"
        backend.copy_file(src.uri, dst_uri)
        return "updated"

    backend.copy_file(src.uri, dst_uri)
    return "new"


def _storage_objects_equal(backend: StorageBackend, src_uri: str, dst_uri: str) -> bool:
    with backend.open_reader(src_uri) as src_reader, backend.open_reader(dst_uri) as dst_reader:
        return _streams_equal(src_reader, dst_reader)


def _streams_equal(left, right) -> bool:
    while True:
        left_chunk = left.read(1024 * 1024)
        right_chunk = right.read(1024 * 1024)
        if left_chunk != right_chunk:
            return False
        if not left_chunk:
            return True


def scan_raw_images(
    raw_root: StoragePath,
    backend: Optional[StorageBackend] = None,
) -> Iterable[StorageFile]:
    backend = backend or backend_for_locations(raw_root)
    yield from backend.list_image_files(backend.normalize_uri(raw_root), IMAGE_EXTS)


def flatten_and_index(
    raw_root: StoragePath,
    flat_out: StoragePath,
    csv_out: StoragePath,
    rehash_all: bool = False,
    keep_duplicates: bool = False,
) -> List[Record]:
    """
    Incrementally updates csv_out and copies new images to flat_out.
    - If size+mtime unchanged and not rehash_all -> skip hashing.
    - Uses sha256 as image_id.
    - Duplicates: if sha already exists and not keep_duplicates -> mark as duplicate and skip copy.
    """
    backend = backend_for_locations(raw_root, flat_out, csv_out)
    raw_root_uri = backend.normalize_uri(raw_root)
    flat_out_uri = backend.normalize_uri(flat_out)
    csv_out_uri = backend.normalize_uri(csv_out)

    rows_by_raw, known_hashes = read_existing_csv(csv_out_uri, backend=backend)
    # rows_by_raw dict with path as key, and csv fields as value
    # known_hashes, set of all hashes

    results: List[Record] = []

    for fp in scan_raw_images(raw_root_uri, backend=backend):
        microscope, parasite = parse_image_metadata(raw_root_uri, fp)

        size_bytes = fp.size_bytes
        mtime_epoch = fp.mtime_epoch
        raw_path_str = fp.uri
        prev = rows_by_raw.get(raw_path_str)  # we check if present in the current csv file
        # fast skip if unchanged
        if prev and not rehash_all:
            try:
                prev_size = int(prev.get("size_bytes", "-1"))
                prev_mtime = int(prev.get("mtime_epoch", "-1"))
            except ValueError:
                prev_size, prev_mtime = -1, -1

            if prev_size == size_bytes and prev_mtime == mtime_epoch:
                row = dict(prev)
                image_id = row.get("image_id", "")
                canonical = (
                    canonical_filename(microscope, parasite, image_id, fp.suffix)
                    if image_id
                    else row.get("canonical_name", "")
                )
                flat_path = (
                    backend.child_uri(flat_out_uri, canonical)
                    if canonical
                    else _resolve_previous_flat_path(row, flat_out_uri, backend)
                )

                status = "skipped_unchanged"
                if canonical and row.get("canonical_name") != canonical:
                    copy_status = copy_storage_if_needed(backend, fp, flat_path)
                    status = "updated"
                elif flat_path and not backend.exists(flat_path):
                    copy_status = copy_storage_if_needed(backend, fp, flat_path)
                    status = copy_status if copy_status != "copied_exists" else "skipped_unchanged"

                row.update(
                    {
                        "microscope": microscope,
                        "parasite": parasite,
                        "raw_path": raw_path_str,
                        "raw_filename": fp.name,
                        "ext": fp.suffix,
                        "size_bytes": str(size_bytes),
                        "mtime_epoch": str(mtime_epoch),
                        "canonical_name": canonical,
                        "flat_path": flat_path if flat_path else row.get("flat_path", ""),
                        "status": status,
                    }
                )
                rows_by_raw[raw_path_str] = row
                results.append(
                    Record(
                        image_id=row.get("image_id", ""),
                        microscope=row.get("microscope", microscope),
                        parasite=row.get("parasite", parasite),
                        raw_path=raw_path_str,
                        raw_filename=fp.name,
                        ext=fp.suffix,
                        size_bytes=size_bytes,
                        mtime_epoch=mtime_epoch,
                        canonical_name=row.get("canonical_name", ""),
                        flat_path=row.get("flat_path", ""),
                        status=status,
                    )
                )
                continue

        image_id = sha256_storage_file(backend, fp)

        canonical = canonical_filename(microscope, parasite, image_id, fp.suffix)
        dst = backend.child_uri(flat_out_uri, canonical)
        
        same_existing_record = prev is not None and prev.get("image_id") == image_id
        if image_id in known_hashes and not keep_duplicates and not same_existing_record:
            row = {
                "image_id": image_id,
                "microscope": microscope,
                "parasite": parasite,
                "raw_path": raw_path_str,
                "raw_filename": fp.name,
                "ext": fp.suffix,
                "size_bytes": str(size_bytes),
                "mtime_epoch": str(mtime_epoch),
                "canonical_name": canonical,
                "flat_path": dst,
                "status": "duplicate",
            }
            rows_by_raw[raw_path_str] = row
            results.append(
                Record(
                    image_id=image_id,
                    microscope=microscope,
                    parasite=parasite,
                    raw_path=raw_path_str,
                    raw_filename=fp.name,
                    ext=fp.suffix,
                    size_bytes=size_bytes,
                    mtime_epoch=mtime_epoch,
                    canonical_name=canonical,
                    flat_path=dst,
                    status="duplicate",
                )
            )
            continue

        copy_status = copy_storage_if_needed(backend, fp, dst)
        known_hashes.add(image_id)

        status = "updated" if prev else copy_status  # "new" or "updated"
        row = {
            "image_id": image_id,
            "microscope": microscope,
            "parasite": parasite,
            "raw_path": raw_path_str,
            "raw_filename": fp.name,
            "ext": fp.suffix,
            "size_bytes": str(size_bytes),
            "mtime_epoch": str(mtime_epoch),
            "canonical_name": canonical,
            "flat_path": dst,
            "status": status,
        }
        rows_by_raw[raw_path_str] = row

        results.append(
            Record(
                image_id=image_id,
                microscope=microscope,
                parasite=parasite,
                raw_path=raw_path_str,
                raw_filename=fp.name,
                ext=fp.suffix,
                size_bytes=size_bytes,
                mtime_epoch=mtime_epoch,
                canonical_name=canonical,
                flat_path=dst,
                status=status,
            )
        )

    write_csv(csv_out_uri, rows_by_raw, backend=backend)
    return results


def _resolve_previous_flat_path(
    prev: Dict[str, str],
    flat_out: StoragePath,
    backend: StorageBackend,
) -> Optional[str]:
    flat_path = prev.get("flat_path", "")
    if flat_path:
        return backend.resolve_reference(flat_path, flat_out)

    canonical_name = prev.get("canonical_name", "")
    if canonical_name:
        return backend.child_uri(flat_out, canonical_name)
    return None


def create_update_db(
    repo_root: Optional[Path],
    rehash_all: bool,
    keep_duplicates: bool,
    storage_root: Optional[str] = None,
    raw_root: Optional[str] = None,
    flat_out: Optional[str] = None,
    csv_out: Optional[str] = None,
) -> None:
    raw_root_ref, flat_out_ref, csv_out_ref = _resolve_db_locations(
        repo_root=repo_root,
        storage_root=storage_root,
        raw_root=raw_root,
        flat_out=flat_out,
        csv_out=csv_out,
    )

    results = flatten_and_index(
        raw_root=raw_root_ref,
        flat_out=flat_out_ref,
        csv_out=csv_out_ref,
        rehash_all=rehash_all,
        keep_duplicates=keep_duplicates,
    )

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print("Repo root:", repo_root if repo_root else "(not used)")
    print("Raw root:", raw_root_ref)
    print("Flat out:", flat_out_ref)
    print("CSV out:", csv_out_ref)
    print("Run summary:", counts)


def _resolve_db_locations(
    repo_root: Optional[Path],
    storage_root: Optional[str],
    raw_root: Optional[str],
    flat_out: Optional[str],
    csv_out: Optional[str],
) -> Tuple[StoragePath, StoragePath, StoragePath]:
    storage_root = storage_root or os.environ.get(STORAGE_ROOT_ENV)
    if storage_root:
        return (
            raw_root or join_storage_uri(storage_root, "raw"),
            flat_out or join_storage_uri(storage_root, "preprocess", "flat_images"),
            csv_out or join_storage_uri(storage_root, "preprocess", "metadata", "master_index.csv"),
        )

    if raw_root and flat_out and csv_out:
        return raw_root, flat_out, csv_out

    if repo_root is None:
        try:
            repo_root = find_repo_root(Path.cwd())
        except RuntimeError as exc:
            if _looks_like_gcp_runtime():
                return (
                    raw_root or join_storage_uri(DEFAULT_GCS_STORAGE_ROOT, "raw"),
                    flat_out or join_storage_uri(DEFAULT_GCS_STORAGE_ROOT, "preprocess", "flat_images"),
                    csv_out or join_storage_uri(
                        DEFAULT_GCS_STORAGE_ROOT,
                        "preprocess",
                        "metadata",
                        "master_index.csv",
                    ),
                )
            raise RuntimeError(
                "Could not locate local data/raw. For GCS, pass "
                "`--storage-root gs://stool-parasite-images` or set "
                f"`{STORAGE_ROOT_ENV}=gs://stool-parasite-images`."
            ) from exc

    raw_root_ref = raw_root or str((repo_root / "data" / "raw").resolve())
    flat_out_ref = flat_out or str((repo_root / "data" / "preprocess" / "flat_images").resolve())
    csv_out_ref = csv_out or str((repo_root / "data" / "preprocess" / "metadata" / "master_index.csv").resolve())
    return raw_root_ref, flat_out_ref, csv_out_ref


def _looks_like_gcp_runtime() -> bool:
    gcp_runtime_env = (
        "K_SERVICE",
        "CLOUD_RUN_JOB",
        "FUNCTION_TARGET",
        "GAE_ENV",
        "GCE_METADATA_HOST",
        "GCE_METADATA_IP",
    )
    return any(os.environ.get(name) for name in gcp_runtime_env)
