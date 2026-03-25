#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Record:
    image_id: str
    microscope: str
    parasite: str
    user: str
    raw_path: str
    raw_filename: str
    ext: str
    size_bytes: int
    mtime_epoch: int
    canonical_name: str
    flat_path: str
    status: str  # "new" | "updated" | "duplicate" | "copied_exists" | "skipped_unchanged"


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


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def parse_triplet(raw_root: Path, file_path: Path) -> Tuple[str, str, str]:
    """
    Extract (microscope, parasite, user) from:
      raw_root/microscope/parasite/user/...
    """
    rel = file_path.relative_to(raw_root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(f"Path does not match <microscope>/<parasite>/<user>/...: {file_path}")
    microscope, parasite, user = parts[0], parts[1], parts[2]
    return microscope, parasite, user


def safe_token(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s.strip())
    return out or "unknown"


def canonical_filename(microscope: str, parasite: str, user: str, image_id: str, ext: str) -> str:
    # short hash keeps names readable
    return f"{safe_token(microscope).lower()}__{safe_token(parasite).lower()}__{safe_token(user).lower()}__{image_id[:12]}{ext.lower()}"

def read_existing_csv(csv_path: Path) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    """
    Returns:
      - by_raw_path: raw_path -> row dict
      - known_hashes: set(image_id)
    """
    if not csv_path.exists():
        return {}, set()

    by_raw_path: Dict[str, Dict[str, str]] = {}
    known_hashes: Set[str] = set()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rp = row.get("raw_path", "")
            if rp:
                by_raw_path[rp] = row
            iid = row.get("image_id", "")
            if iid:
                known_hashes.add(iid)
    return by_raw_path, known_hashes


def write_csv(csv_path: Path, rows_by_raw: Dict[str, Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "microscope",
        "parasite",
        "user",
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
    rows.sort(key=lambda r: (r.get("microscope", ""), r.get("parasite", ""), r.get("user", ""), r.get("raw_path", "")))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def copy_if_needed(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "copied_exists"
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            b = fsrc.read(1024 * 1024)
            if not b:
                break
            fdst.write(b)
    return "new"


def scan_raw_images(raw_root: Path) -> Iterable[Path]:
    for p in raw_root.rglob("*"):
        if is_image_file(p):
            yield p


def flatten_and_index(
    raw_root: Path,
    flat_out: Path,
    csv_out: Path,
    rehash_all: bool = False,
    keep_duplicates: bool = False,
) -> List[Record]:
    """
    Incrementally updates csv_out and copies new images to flat_out.
    - If size+mtime unchanged and not rehash_all -> skip hashing.
    - Uses sha256 as image_id.
    - Duplicates: if sha already exists and not keep_duplicates -> mark as duplicate and skip copy.
    """
    rows_by_raw, known_hashes = read_existing_csv(csv_out)
    # rows_by_raw dict with path as key, and csv fields as value
    # known_hashes, set of all hashes

    results: List[Record] = []

    for fp in scan_raw_images(raw_root):
        microscope, parasite, user = parse_triplet(raw_root, fp)

        st = fp.stat()
        size_bytes = int(st.st_size)
        mtime_epoch = int(st.st_mtime)
        raw_path_str = str(fp.resolve()) # this gives the path to raw folder
        prev = rows_by_raw.get(raw_path_str) # we check if present in the current csv file
        #############################################################################################
        print(prev) 
        # fast skip if unchanged
        if prev and not rehash_all:
            try:
                prev_size = int(prev.get("size_bytes", "-1"))
                prev_mtime = int(prev.get("mtime_epoch", "-1"))
            except ValueError:
                prev_size, prev_mtime = -1, -1

            if prev_size == size_bytes and prev_mtime == mtime_epoch:
                results.append(
                    Record(
                        image_id=prev.get("image_id", ""),
                        microscope=prev.get("microscope", microscope),
                        parasite=prev.get("parasite", parasite),
                        user=prev.get("user", user),
                        raw_path=raw_path_str,
                        raw_filename=fp.name,
                        ext=fp.suffix.lower(),
                        size_bytes=size_bytes,
                        mtime_epoch=mtime_epoch,
                        canonical_name=prev.get("canonical_name", ""),
                        flat_path=prev.get("flat_path", ""),
                        status="skipped_unchanged",
                    )
                )
                continue

        image_id = sha256_file(fp)

        canonical = canonical_filename(microscope, parasite, user, image_id, fp.suffix)
        dst = flat_out / canonical
        
        if image_id in known_hashes and not keep_duplicates:
            row = {
                "image_id": image_id,
                "microscope": microscope,
                "parasite": parasite,
                "user": user,
                "raw_path": raw_path_str,
                "raw_filename": fp.name,
                "ext": fp.suffix.lower(),
                "size_bytes": str(size_bytes),
                "mtime_epoch": str(mtime_epoch),
                "canonical_name": canonical,
                "flat_path": str(dst.resolve()),
                "status": "duplicate",
            }
            rows_by_raw[raw_path_str] = row
            results.append(
                Record(
                    image_id=image_id,
                    microscope=microscope,
                    parasite=parasite,
                    user=user,
                    raw_path=raw_path_str,
                    raw_filename=fp.name,
                    ext=fp.suffix.lower(),
                    size_bytes=size_bytes,
                    mtime_epoch=mtime_epoch,
                    canonical_name=canonical,
                    flat_path=str(dst.resolve()),
                    status="duplicate",
                )
            )
            continue

        copy_status = copy_if_needed(fp, dst)
        known_hashes.add(image_id)

        status = "updated" if prev else copy_status  # "new" or "updated"
        row = {
            "image_id": image_id,
            "microscope": microscope,
            "parasite": parasite,
            "user": user,
            "raw_path": raw_path_str,
            "raw_filename": fp.name,
            "ext": fp.suffix.lower(),
            "size_bytes": str(size_bytes),
            "mtime_epoch": str(mtime_epoch),
            "canonical_name": canonical,
            "flat_path": str(dst.resolve()),
            "status": status,
        }
        rows_by_raw[raw_path_str] = row

        results.append(
            Record(
                image_id=image_id,
                microscope=microscope,
                parasite=parasite,
                user=user,
                raw_path=raw_path_str,
                raw_filename=fp.name,
                ext=fp.suffix.lower(),
                size_bytes=size_bytes,
                mtime_epoch=mtime_epoch,
                canonical_name=canonical,
                flat_path=str(dst.resolve()),
                status=status,
            )
        )

    write_csv(csv_out, rows_by_raw)
    return results


def create_update_db(repo_root, rehash_all, keep_duplicates):
    raw_root = repo_root / "data" / "raw"
    preprocess_root = repo_root / "data" / "preprocess"
    flat_out = preprocess_root / "flat_images"
    csv_out = preprocess_root / "metadata" / "master_index.csv"

    results = flatten_and_index(
        raw_root=raw_root,
        flat_out=flat_out,
        csv_out=csv_out,
        rehash_all=rehash_all,
        keep_duplicates=keep_duplicates,
    )

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print("Repo root:", repo_root)
    print("Raw root:", raw_root)
    print("Flat out:", flat_out)
    print("CSV out:", csv_out)
    print("Run summary:", counts)
