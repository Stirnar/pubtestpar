from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import re
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class LabelBox:
    id: str
    line: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    raw: str

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "line": self.line,
            "class_name": self.class_name,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class LabelParseResult:
    boxes: list[LabelBox]
    errors: list[str]


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    label_filename: str | None
    label_status: str
    box_count: int
    classes: list[str]
    errors: list[str]

    def as_json(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "label_filename": self.label_filename,
            "label_status": self.label_status,
            "box_count": self.box_count,
            "classes": self.classes,
            "errors": self.errors,
        }


class CanonicalLabelIndex:
    def __init__(self, images_dir: Path, labels_dir: Path) -> None:
        self.images_dir = images_dir.expanduser().resolve()
        self.labels_dir = labels_dir.expanduser().resolve()
        self.images: list[ImageRecord] = []
        self.boxes_by_filename: dict[str, LabelParseResult] = {}
        self.orphan_labels: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        image_paths = (
            sorted(
                path
                for path in self.images_dir.glob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if self.images_dir.exists()
            else []
        )
        label_paths = (
            sorted(path for path in self.labels_dir.glob("*.txt") if path.is_file())
            if self.labels_dir.exists()
            else []
        )
        label_by_stem = {path.stem: path for path in label_paths}
        image_stems = {path.stem for path in image_paths}

        self.images = []
        self.boxes_by_filename = {}

        for image_path in image_paths:
            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                result = LabelParseResult(boxes=[], errors=[])
                status = "missing"
                label_filename = None
            else:
                result = parse_yolo_label_file(label_path)
                if result.errors:
                    status = "invalid"
                elif not result.boxes:
                    status = "empty"
                else:
                    status = "ok"
                label_filename = label_path.name

            classes = sorted({box.class_name for box in result.boxes})
            record = ImageRecord(
                filename=image_path.name,
                label_filename=label_filename,
                label_status=status,
                box_count=len(result.boxes),
                classes=classes,
                errors=result.errors,
            )
            self.images.append(record)
            self.boxes_by_filename[image_path.name] = result

        self.orphan_labels = sorted(path.name for path in label_paths if path.stem not in image_stems)

    def summary(self) -> dict[str, Any]:
        counts = {"ok": 0, "empty": 0, "missing": 0, "invalid": 0}
        total_boxes = 0
        classes: set[str] = set()
        for image in self.images:
            counts[image.label_status] += 1
            total_boxes += image.box_count
            classes.update(image.classes)
        return {
            "total_images": len(self.images),
            "total_boxes": total_boxes,
            "status_counts": counts,
            "classes": sorted(classes),
            "orphan_labels": len(self.orphan_labels),
            "images_dir": str(self.images_dir),
            "labels_dir": str(self.labels_dir),
            "images_dir_exists": self.images_dir.exists(),
            "labels_dir_exists": self.labels_dir.exists(),
        }


def parse_yolo_label_file(path: Path) -> LabelParseResult:
    boxes: list[LabelBox] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return LabelParseResult(boxes=[], errors=[f"{path.name}: cannot decode as UTF-8"])

    for line_number, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(parts)}")
            continue

        class_name = parts[0]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", class_name):
            errors.append(f"line {line_number}: invalid class token {class_name!r}")
            continue

        try:
            x_center, y_center, width, height = [float(token) for token in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: coordinates must be numeric")
            continue

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            errors.append(f"line {line_number}: center coordinates must be in [0, 1]")
            continue
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors.append(f"line {line_number}: width and height must be in (0, 1]")
            continue
        if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1 + 1e-6:
            errors.append(f"line {line_number}: box extends outside x bounds")
            continue
        if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1 + 1e-6:
            errors.append(f"line {line_number}: box extends outside y bounds")
            continue

        boxes.append(
            LabelBox(
                id=f"L{line_number}",
                line=line_number,
                class_name=class_name,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                raw=stripped,
            )
        )

    return LabelParseResult(boxes=boxes, errors=errors)


def _safe_child(base: Path, relative_name: str) -> Path | None:
    clean_name = posixpath.basename(unquote(relative_name))
    candidate = (base / clean_name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def make_handler(index: CanonicalLabelIndex) -> type[BaseHTTPRequestHandler]:
    class CanonicalLabelViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                self._send_html(HTML_TEMPLATE)
                return

            if path == "/api/images":
                index.refresh()
                self._send_json(
                    {
                        "images": [image.as_json() for image in index.images],
                        "summary": index.summary(),
                    }
                )
                return

            if path == "/api/orphan-labels":
                index.refresh()
                self._send_json({"orphan_labels": index.orphan_labels})
                return

            if path.startswith("/api/labels/"):
                filename = unquote(path.removeprefix("/api/labels/"))
                result = index.boxes_by_filename.get(posixpath.basename(filename))
                if result is None:
                    self._send_json({"boxes": [], "errors": ["image not indexed"]}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(
                    {
                        "boxes": [box.as_json() for box in result.boxes],
                        "errors": result.errors,
                    }
                )
                return

            if path.startswith("/api/image/"):
                filename = path.removeprefix("/api/image/")
                image_path = _safe_child(index.images_dir, filename)
                if image_path is None or not image_path.is_file():
                    self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
                    return
                self._send_file(image_path)
                return

            self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return CanonicalLabelViewerHandler


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def run_server(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else repo_root_from_here()
    images_dir = Path(args.images_dir).expanduser() if args.images_dir else repo_root / "data" / "preprocess" / "flat_images"
    labels_dir = Path(args.labels_dir).expanduser() if args.labels_dir else repo_root / "data" / "labels" / "canonical"
    if not images_dir.is_absolute():
        images_dir = repo_root / images_dir
    if not labels_dir.is_absolute():
        labels_dir = repo_root / labels_dir

    index = CanonicalLabelIndex(images_dir=images_dir, labels_dir=labels_dir)
    handler = make_handler(index)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"

    print(f"Canonical label viewer: {url}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(
        "Indexed "
        f"{index.summary()['total_images']} images, "
        f"{index.summary()['total_boxes']} boxes, "
        f"{index.summary()['orphan_labels']} label files without matching images."
    )

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping canonical label viewer.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only browser viewer for flat images and canonical YOLO labels."
    )
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to this script's parent repo.")
    parser.add_argument("--images-dir", default=None, help="Flat images directory. Defaults to data/preprocess/flat_images.")
    parser.add_argument("--labels-dir", default=None, help="Canonical YOLO label directory. Defaults to data/labels/canonical.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=5055, help="Server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Canonical Label Viewer</title>
<style>
:root {
  --bg: #101214;
  --panel: #181d22;
  --panel-2: #20262d;
  --line: #33404b;
  --text: #ecf1f4;
  --muted: #9aa8b2;
  --soft: #6c7882;
  --blue: #5aa9ff;
  --green: #42d392;
  --red: #ff5d73;
  --yellow: #f7bd45;
  --cyan: #4bd6d6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input, select {
  font: inherit;
}
.app {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr) 330px;
  grid-template-rows: 54px minmax(0, 1fr);
  height: 100vh;
}
.top {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 14px;
  background: var(--panel);
  border-bottom: 1px solid var(--line);
}
.brand {
  font-weight: 800;
  letter-spacing: .02em;
  color: var(--text);
  white-space: nowrap;
}
.stats {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.pill {
  border: 1px solid var(--line);
  background: var(--panel-2);
  color: var(--muted);
  border-radius: 6px;
  padding: 5px 8px;
  font-size: 12px;
  white-space: nowrap;
}
.pill.ok { color: var(--green); }
.pill.empty { color: var(--yellow); }
.pill.missing { color: var(--soft); }
.pill.invalid { color: var(--red); }
.grow { flex: 1; }
.control {
  background: var(--panel-2);
  border: 1px solid var(--line);
  color: var(--text);
  border-radius: 6px;
  height: 34px;
  padding: 0 10px;
  min-width: 0;
}
.btn {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-2);
  color: var(--text);
  cursor: pointer;
  padding: 0 10px;
}
.btn:hover { border-color: var(--blue); }
.left, .right {
  min-height: 0;
  overflow: hidden;
  background: var(--panel);
}
.left {
  border-right: 1px solid var(--line);
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
}
.filters {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 110px;
  gap: 8px;
  padding: 10px;
  border-bottom: 1px solid var(--line);
}
.list {
  overflow: auto;
}
.item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  padding: 9px 10px;
  border-left: 4px solid transparent;
  border-bottom: 1px solid rgba(51, 64, 75, .55);
  cursor: pointer;
}
.item:hover { background: rgba(255,255,255,.035); }
.item.active {
  background: rgba(90, 169, 255, .12);
  border-left-color: var(--blue);
}
.name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}
.meta {
  color: var(--muted);
  font-size: 12px;
  margin-top: 3px;
}
.tag {
  align-self: start;
  border-radius: 5px;
  padding: 3px 6px;
  font-size: 11px;
  border: 1px solid var(--line);
}
.tag.ok { color: var(--green); border-color: rgba(66, 211, 146, .55); }
.tag.empty { color: var(--yellow); border-color: rgba(247, 189, 69, .55); }
.tag.missing { color: var(--soft); }
.tag.invalid { color: var(--red); border-color: rgba(255, 93, 115, .55); }
.stage {
  min-width: 0;
  min-height: 0;
  position: relative;
  overflow: hidden;
  background: #090b0d;
}
canvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
}
canvas.dragging { cursor: grabbing; }
.empty-state {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
}
.hud {
  position: absolute;
  left: 12px;
  top: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  border: 1px solid rgba(51, 64, 75, .9);
  background: rgba(24, 29, 34, .9);
  border-radius: 7px;
}
.hud .btn {
  width: 34px;
  padding: 0;
  height: 28px;
}
.zoom-text {
  color: var(--muted);
  font-size: 12px;
  min-width: 48px;
  text-align: center;
}
.right {
  border-left: 1px solid var(--line);
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
}
.side-section {
  padding: 10px;
  border-bottom: 1px solid var(--line);
}
.title {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 700;
  font-size: 11px;
  margin-bottom: 8px;
}
.classes {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.class-chip {
  border: 1px solid var(--line);
  background: var(--panel-2);
  color: var(--text);
  border-radius: 6px;
  padding: 5px 8px;
  cursor: pointer;
  font-size: 12px;
}
.class-chip.off {
  color: var(--soft);
  text-decoration: line-through;
  opacity: .65;
}
.box-list {
  overflow: auto;
}
.box-item {
  border-bottom: 1px solid rgba(51, 64, 75, .6);
  padding: 9px 10px;
  cursor: pointer;
}
.box-item:hover { background: rgba(255,255,255,.035); }
.box-item.active {
  background: rgba(90, 169, 255, .12);
}
.box-head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
}
.line-no {
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.coords {
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  margin-top: 5px;
  overflow-wrap: anywhere;
}
.error {
  color: var(--red);
  border-top: 1px solid rgba(255, 93, 115, .25);
  padding-top: 8px;
  margin-top: 8px;
  font-size: 12px;
}
.toggle-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.toggle-row label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 980px) {
  .app {
    grid-template-columns: 260px minmax(0, 1fr);
  }
  .right {
    display: none;
  }
}
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="brand">Canonical Label Viewer</div>
    <div class="stats" id="stats"></div>
    <div class="grow"></div>
    <button class="btn" id="prevBtn">Prev</button>
    <button class="btn" id="nextBtn">Next</button>
    <button class="btn" id="reloadBtn">Reload</button>
  </header>

  <aside class="left">
    <div class="filters">
      <input class="control" id="search" placeholder="Search filename or class">
      <select class="control" id="statusFilter">
        <option value="all">All</option>
        <option value="ok">With boxes</option>
        <option value="empty">Empty</option>
        <option value="missing">Missing label</option>
        <option value="invalid">Invalid label</option>
      </select>
    </div>
    <div class="list" id="imageList"></div>
  </aside>

  <main class="stage" id="stage">
    <canvas id="canvas"></canvas>
    <div class="empty-state" id="emptyState">No image selected</div>
    <div class="hud">
      <button class="btn" id="zoomOutBtn">-</button>
      <div class="zoom-text" id="zoomText">100%</div>
      <button class="btn" id="zoomInBtn">+</button>
      <button class="btn" id="fitBtn">Fit</button>
    </div>
  </main>

  <aside class="right">
    <div class="side-section">
      <div class="title">Display</div>
      <div class="toggle-row">
        <label><input type="checkbox" id="showBoxes" checked> Boxes</label>
        <label><input type="checkbox" id="showLabels" checked> Labels</label>
      </div>
    </div>
    <div class="side-section">
      <div class="title">Classes</div>
      <div class="classes" id="classList"></div>
    </div>
    <div class="box-list" id="boxList"></div>
  </aside>
</div>

<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const stage = document.getElementById("stage");
const emptyState = document.getElementById("emptyState");
const state = {
  images: [],
  filtered: [],
  summary: null,
  current: -1,
  boxes: [],
  errors: [],
  imageElement: null,
  imageLoaded: false,
  scale: 1,
  baseScale: 1,
  panX: 0,
  panY: 0,
  hiddenClasses: new Set(),
  activeBox: null,
  dragging: false,
  dragStart: null
};
const colors = ["#42d392", "#5aa9ff", "#f7bd45", "#ff5d73", "#4bd6d6", "#b889ff", "#ff9d58"];

async function loadIndex(keepSelection = false) {
  const selectedName = keepSelection && state.filtered[state.current]
    ? state.filtered[state.current].filename
    : null;
  const response = await fetch("/api/images");
  const data = await response.json();
  state.images = data.images;
  state.summary = data.summary;
  renderStats();
  applyFilters();
  const nextIndex = selectedName
    ? state.filtered.findIndex(item => item.filename === selectedName)
    : 0;
  if (state.filtered.length) {
    await selectImage(Math.max(0, nextIndex));
  } else {
    state.current = -1;
    state.boxes = [];
    state.errors = [];
    state.imageElement = null;
    draw();
    renderBoxes();
  }
}

function renderStats() {
  const summary = state.summary;
  if (!summary) return;
  const counts = summary.status_counts;
  document.getElementById("stats").innerHTML = [
    `<span class="pill">${summary.total_images} images</span>`,
    `<span class="pill ok">${counts.ok} with boxes</span>`,
    `<span class="pill empty">${counts.empty} empty</span>`,
    `<span class="pill missing">${counts.missing} missing</span>`,
    `<span class="pill invalid">${counts.invalid} invalid</span>`,
    `<span class="pill">${summary.total_boxes} boxes</span>`,
    `<span class="pill">${summary.orphan_labels} orphan labels</span>`,
    summary.images_dir_exists ? "" : `<span class="pill invalid">images dir missing</span>`,
    summary.labels_dir_exists ? "" : `<span class="pill invalid">labels dir missing</span>`
  ].join("");
}

function applyFilters() {
  const term = document.getElementById("search").value.trim().toLowerCase();
  const status = document.getElementById("statusFilter").value;
  state.filtered = state.images.filter(item => {
    const haystack = `${item.filename} ${item.classes.join(" ")}`.toLowerCase();
    return (status === "all" || item.label_status === status) && (!term || haystack.includes(term));
  });
  renderImageList();
}

function renderImageList() {
  const list = document.getElementById("imageList");
  list.innerHTML = state.filtered.map((item, index) => {
    const classes = item.classes.length ? item.classes.join(", ") : "no boxes";
    return `<div class="item ${index === state.current ? "active" : ""}" data-index="${index}">
      <div>
        <div class="name">${escapeHtml(item.filename)}</div>
        <div class="meta">${item.box_count} boxes | ${escapeHtml(classes)}</div>
      </div>
      <div class="tag ${item.label_status}">${item.label_status}</div>
    </div>`;
  }).join("");
  list.querySelectorAll(".item").forEach(node => {
    node.addEventListener("click", () => selectImage(Number(node.dataset.index)));
  });
}

async function selectImage(index) {
  if (index < 0 || index >= state.filtered.length) return;
  state.current = index;
  state.activeBox = null;
  renderImageList();
  const item = state.filtered[state.current];
  const labelResponse = await fetch(`/api/labels/${encodeURIComponent(item.filename)}`);
  const labelData = await labelResponse.json();
  state.boxes = labelData.boxes || [];
  state.errors = labelData.errors || [];
  state.imageLoaded = false;
  emptyState.style.display = "none";

  const image = new Image();
  image.onload = () => {
    state.imageElement = image;
    state.imageLoaded = true;
    fitImage();
    renderBoxes();
    renderClasses();
    draw();
  };
  image.onerror = () => {
    state.imageElement = null;
    state.imageLoaded = false;
    emptyState.style.display = "flex";
    emptyState.textContent = "Image failed to load";
    renderBoxes();
    draw();
  };
  image.src = `/api/image/${encodeURIComponent(item.filename)}`;
}

function renderClasses() {
  const allClasses = Array.from(new Set(state.boxes.map(box => box.class_name))).sort();
  document.getElementById("classList").innerHTML = allClasses.map(className => {
    const off = state.hiddenClasses.has(className);
    return `<button class="class-chip ${off ? "off" : ""}" data-class="${escapeHtml(className)}">${escapeHtml(className)}</button>`;
  }).join("");
  document.querySelectorAll(".class-chip").forEach(node => {
    node.addEventListener("click", () => {
      const className = node.dataset.class;
      if (state.hiddenClasses.has(className)) state.hiddenClasses.delete(className);
      else state.hiddenClasses.add(className);
      renderClasses();
      draw();
    });
  });
}

function renderBoxes() {
  const item = state.filtered[state.current];
  const title = item ? `<div class="side-section"><div class="title">${escapeHtml(item.filename)}</div></div>` : "";
  const boxes = state.boxes.map(box => {
    const active = state.activeBox === box.id;
    return `<div class="box-item ${active ? "active" : ""}" data-box="${box.id}">
      <div class="box-head">
        <strong>${escapeHtml(box.class_name)}</strong>
        <span class="line-no">line ${box.line}</span>
      </div>
      <div class="coords">${box.x_center.toFixed(6)} ${box.y_center.toFixed(6)} ${box.width.toFixed(6)} ${box.height.toFixed(6)}</div>
    </div>`;
  }).join("");
  const errors = state.errors.map(error => `<div class="error">${escapeHtml(error)}</div>`).join("");
  document.getElementById("boxList").innerHTML = title + boxes + errors;
  document.querySelectorAll(".box-item").forEach(node => {
    node.addEventListener("click", () => {
      state.activeBox = node.dataset.box;
      renderBoxes();
      draw();
    });
  });
}

function resizeCanvas() {
  const rect = stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function fitImage() {
  if (!state.imageElement) return;
  const rect = stage.getBoundingClientRect();
  state.baseScale = Math.min(rect.width / state.imageElement.naturalWidth, rect.height / state.imageElement.naturalHeight) * 0.94;
  state.scale = state.baseScale;
  state.panX = (rect.width - state.imageElement.naturalWidth * state.scale) / 2;
  state.panY = (rect.height - state.imageElement.naturalHeight * state.scale) / 2;
  updateZoomText();
}

function zoomAt(factor, originX, originY) {
  if (!state.imageElement) return;
  const beforeX = (originX - state.panX) / state.scale;
  const beforeY = (originY - state.panY) / state.scale;
  state.scale = clamp(state.scale * factor, state.baseScale * 0.2, state.baseScale * 20);
  state.panX = originX - beforeX * state.scale;
  state.panY = originY - beforeY * state.scale;
  updateZoomText();
  draw();
}

function updateZoomText() {
  document.getElementById("zoomText").textContent = `${Math.round(state.scale / state.baseScale * 100)}%`;
}

function draw() {
  const rect = stage.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!state.imageElement || !state.imageLoaded) return;

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(
    state.imageElement,
    state.panX,
    state.panY,
    state.imageElement.naturalWidth * state.scale,
    state.imageElement.naturalHeight * state.scale
  );

  if (!document.getElementById("showBoxes").checked) return;

  const visibleBoxes = state.boxes.filter(box => !state.hiddenClasses.has(box.class_name));
  visibleBoxes.forEach((box, index) => drawBox(box, index));
}

function drawBox(box, index) {
  const image = state.imageElement;
  const color = colorForClass(box.class_name, index);
  const x = state.panX + (box.x_center - box.width / 2) * image.naturalWidth * state.scale;
  const y = state.panY + (box.y_center - box.height / 2) * image.naturalHeight * state.scale;
  const w = box.width * image.naturalWidth * state.scale;
  const h = box.height * image.naturalHeight * state.scale;
  const active = state.activeBox === box.id;

  ctx.save();
  ctx.lineWidth = active ? 4 : 2;
  ctx.strokeStyle = color;
  ctx.fillStyle = color + "22";
  ctx.strokeRect(x, y, w, h);
  ctx.fillRect(x, y, w, h);

  if (document.getElementById("showLabels").checked) {
    const label = `${box.class_name} L${box.line}`;
    ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
    const metrics = ctx.measureText(label);
    const labelW = metrics.width + 10;
    const labelH = 20;
    const labelY = y - labelH >= 0 ? y - labelH : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, labelW, labelH);
    ctx.fillStyle = "#071014";
    ctx.fillText(label, x + 5, labelY + 14);
  }
  ctx.restore();
}

function colorForClass(className, fallbackIndex) {
  let hash = 0;
  for (let i = 0; i < className.length; i++) hash = (hash * 31 + className.charCodeAt(i)) >>> 0;
  return colors[(hash || fallbackIndex) % colors.length];
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[char]));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

document.getElementById("search").addEventListener("input", applyFilters);
document.getElementById("statusFilter").addEventListener("change", applyFilters);
document.getElementById("reloadBtn").addEventListener("click", () => loadIndex(true));
document.getElementById("prevBtn").addEventListener("click", () => selectImage(state.current - 1));
document.getElementById("nextBtn").addEventListener("click", () => selectImage(state.current + 1));
document.getElementById("fitBtn").addEventListener("click", () => { fitImage(); draw(); });
document.getElementById("zoomOutBtn").addEventListener("click", () => {
  const rect = stage.getBoundingClientRect();
  zoomAt(1 / 1.25, rect.width / 2, rect.height / 2);
});
document.getElementById("zoomInBtn").addEventListener("click", () => {
  const rect = stage.getBoundingClientRect();
  zoomAt(1.25, rect.width / 2, rect.height / 2);
});
document.getElementById("showBoxes").addEventListener("change", draw);
document.getElementById("showLabels").addEventListener("change", draw);

canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  zoomAt(factor, event.clientX - rect.left, event.clientY - rect.top);
}, { passive: false });

canvas.addEventListener("mousedown", event => {
  state.dragging = true;
  canvas.classList.add("dragging");
  state.dragStart = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
});
window.addEventListener("mouseup", () => {
  state.dragging = false;
  canvas.classList.remove("dragging");
});
window.addEventListener("mousemove", event => {
  if (!state.dragging || !state.dragStart) return;
  state.panX = state.dragStart.panX + event.clientX - state.dragStart.x;
  state.panY = state.dragStart.panY + event.clientY - state.dragStart.y;
  draw();
});
window.addEventListener("keydown", event => {
  if (event.key === "ArrowRight") selectImage(state.current + 1);
  if (event.key === "ArrowLeft") selectImage(state.current - 1);
});
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
loadIndex();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    run_server(parse_args())
