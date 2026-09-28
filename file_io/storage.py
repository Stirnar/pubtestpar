from __future__ import annotations

import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Optional, Union

StoragePath = Union[str, Path]


@dataclass(frozen=True)
class StorageFile:
    uri: str
    relative_path: PurePosixPath
    name: str
    suffix: str
    size_bytes: int
    mtime_epoch: int


@dataclass(frozen=True)
class GcsUri:
    bucket: str
    object_name: str = ""

    def as_uri(self) -> str:
        if self.object_name:
            return f"gs://{self.bucket}/{self.object_name}"
        return f"gs://{self.bucket}"


def is_gcs_uri(value: StoragePath) -> bool:
    return str(value).startswith("gs://")


def storage_scheme(value: StoragePath) -> str:
    return "gcs" if is_gcs_uri(value) else "local"


def parse_gcs_uri(uri: StoragePath) -> GcsUri:
    text = str(uri)
    if not text.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {text!r}")
    without_scheme = text[5:]
    bucket, separator, object_name = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Expected a bucket name in GCS URI, got {text!r}")
    return GcsUri(bucket=bucket, object_name=object_name.strip("/"))


def join_storage_uri(root: StoragePath, *parts: str) -> str:
    if is_gcs_uri(root):
        parsed = parse_gcs_uri(root)
        clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
        object_parts = [parsed.object_name] if parsed.object_name else []
        object_parts.extend(clean_parts)
        return GcsUri(parsed.bucket, "/".join(object_parts)).as_uri()

    path = Path(root).expanduser()
    for part in parts:
        if part:
            path = path / part
    return str(path)


def backend_for_locations(*locations: StoragePath) -> "StorageBackend":
    schemes = {storage_scheme(location) for location in locations if location is not None}
    if len(schemes) > 1:
        raise ValueError(
            "Storage locations must all use the same backend. "
            "Pass all local paths or all gs:// URIs."
        )
    scheme = next(iter(schemes), "local")
    if scheme == "gcs":
        return GcsStorageBackend()
    return LocalStorageBackend()


class StorageBackend:
    scheme = "base"

    def normalize_uri(self, uri: StoragePath) -> str:
        raise NotImplementedError

    def child_uri(self, root: StoragePath, *parts: str) -> str:
        raise NotImplementedError

    def list_image_files(self, root: StoragePath, suffixes: set[str]) -> Iterable[StorageFile]:
        raise NotImplementedError

    def list_files(self, root: StoragePath) -> Iterable[StorageFile]:
        raise NotImplementedError

    def exists(self, uri: StoragePath) -> bool:
        raise NotImplementedError

    def read_bytes(self, uri: StoragePath) -> bytes:
        with self.open_reader(uri) as reader:
            return reader.read()

    def write_bytes(
        self,
        uri: StoragePath,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        with self.open_writer(uri, content_type=content_type) as writer:
            writer.write(data)

    def open_reader(self, uri: StoragePath) -> BinaryIO:
        raise NotImplementedError

    def open_writer(self, uri: StoragePath, content_type: Optional[str] = None) -> BinaryIO:
        raise NotImplementedError

    def copy_file(self, src_uri: StoragePath, dst_uri: StoragePath) -> None:
        with self.open_reader(src_uri) as reader, self.open_writer(
            dst_uri,
            content_type=mimetypes.guess_type(str(dst_uri))[0],
        ) as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)

    def resolve_reference(self, value: str, root: StoragePath) -> str:
        if self.is_absolute_reference(value):
            return self.normalize_uri(value)
        return self.child_uri(root, value)

    def is_absolute_reference(self, value: str) -> bool:
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    scheme = "local"

    def normalize_uri(self, uri: StoragePath) -> str:
        return str(Path(uri).expanduser().resolve())

    def child_uri(self, root: StoragePath, *parts: str) -> str:
        path = Path(root).expanduser()
        for part in parts:
            if part:
                path = path / part
        return str(path.resolve())

    def list_image_files(self, root: StoragePath, suffixes: set[str]) -> Iterable[StorageFile]:
        for file in self.list_files(root):
            if file.suffix not in suffixes:
                continue
            yield file

    def list_files(self, root: StoragePath) -> Iterable[StorageFile]:
        root_path = Path(root).expanduser().resolve()
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            relative_path = PurePosixPath(path.relative_to(root_path).as_posix())
            yield StorageFile(
                uri=str(path.resolve()),
                relative_path=relative_path,
                name=path.name,
                suffix=path.suffix.lower(),
                size_bytes=int(stat.st_size),
                mtime_epoch=int(stat.st_mtime),
            )

    def exists(self, uri: StoragePath) -> bool:
        return Path(uri).expanduser().exists()

    def open_reader(self, uri: StoragePath) -> BinaryIO:
        return Path(uri).expanduser().open("rb")

    def open_writer(self, uri: StoragePath, content_type: Optional[str] = None) -> BinaryIO:
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("wb")

    def is_absolute_reference(self, value: str) -> bool:
        return Path(value).expanduser().is_absolute()


class GcsStorageBackend(StorageBackend):
    scheme = "gcs"

    def __init__(self) -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "GCS support requires the google-cloud-storage package. "
                "Install it with `pip install google-cloud-storage`."
            ) from exc

        self._client = storage.Client()

    def normalize_uri(self, uri: StoragePath) -> str:
        return parse_gcs_uri(uri).as_uri()

    def child_uri(self, root: StoragePath, *parts: str) -> str:
        return join_storage_uri(root, *parts)

    def list_image_files(self, root: StoragePath, suffixes: set[str]) -> Iterable[StorageFile]:
        for file in self.list_files(root):
            if file.suffix not in suffixes:
                continue
            yield file

    def list_files(self, root: StoragePath) -> Iterable[StorageFile]:
        parsed = parse_gcs_uri(root)
        prefix = f"{parsed.object_name}/" if parsed.object_name else ""
        for blob in self._client.list_blobs(parsed.bucket, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            relative_name = blob.name[len(prefix):] if prefix else blob.name
            if not relative_name:
                continue
            relative_path = PurePosixPath(relative_name)
            suffix = relative_path.suffix.lower()
            updated = blob.updated
            yield StorageFile(
                uri=GcsUri(parsed.bucket, blob.name).as_uri(),
                relative_path=relative_path,
                name=relative_path.name,
                suffix=suffix,
                size_bytes=int(blob.size or 0),
                mtime_epoch=int(updated.timestamp()) if updated else 0,
            )

    def exists(self, uri: StoragePath) -> bool:
        return self._blob(uri).exists(client=self._client)

    def read_bytes(self, uri: StoragePath) -> bytes:
        return self._blob(uri).download_as_bytes(client=self._client)

    def write_bytes(
        self,
        uri: StoragePath,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        self._blob(uri).upload_from_string(
            data,
            content_type=content_type or "application/octet-stream",
            client=self._client,
        )

    def open_reader(self, uri: StoragePath) -> BinaryIO:
        return self._blob(uri).open("rb")

    def open_writer(self, uri: StoragePath, content_type: Optional[str] = None) -> BinaryIO:
        kwargs = {}
        guessed_content_type = content_type or mimetypes.guess_type(str(uri))[0]
        if guessed_content_type:
            kwargs["content_type"] = guessed_content_type
        return self._blob(uri).open("wb", **kwargs)

    def is_absolute_reference(self, value: str) -> bool:
        return is_gcs_uri(value)

    def _blob(self, uri: StoragePath):
        parsed = parse_gcs_uri(uri)
        if not parsed.object_name:
            raise ValueError(f"Expected a GCS object URI, got {str(uri)!r}")
        bucket = self._client.bucket(parsed.bucket)
        return bucket.blob(parsed.object_name)
