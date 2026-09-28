from __future__ import annotations

import json
import shutil
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PredictionBundleResult:
    output_dir: Path
    predictions_path: Path
    total_images: int
    total_boxes: int
    uploaded_dir: str | None = None


def write_prediction_bundle(
    model: Any,
    image_paths: Sequence[Path],
    output_dir: Path,
    model_path: str,
    data_path: str | None,
    split: str,
    class_names: Sequence[str],
    imgsz: int,
    min_conf: float,
    iou: float,
    device: str | None,
    overwrite: bool = False,
) -> PredictionBundleResult:
    output_dir = output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    predictions_path = output_dir / "predictions_viewer.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Prediction viewer bundle already exists: {output_dir}. Pass --overwrite.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    names = list(class_names) or _normalize_names(getattr(model, "names", {}))
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, 1):
        results = model.predict(
            source=str(image_path),
            conf=min_conf,
            iou=iou,
            imgsz=imgsz,
            save=False,
            save_txt=False,
            verbose=False,
            **({"device": device} if device else {}),
        )
        if not results:
            continue
        result = results[0]
        image_bytes, width, height = _encode_result_image(getattr(result, "orig_img", None))
        display_filename = _display_filename(image_path)
        (images_dir / display_filename).write_bytes(image_bytes)

        image_boxes = _boxes_from_result(result, names)
        images.append(
            {
                "id": image_id,
                "source_filename": image_path.name,
                "source_path": str(image_path),
                "display_filename": display_filename,
                "width": width,
                "height": height,
                "num_boxes": len(image_boxes),
            }
        )
        for box in image_boxes:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    **box,
                }
            )
            annotation_id += 1

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model": model_path,
            "data": data_path,
            "split": split,
            "imgsz": imgsz,
            "min_conf": min_conf,
            "iou": iou,
            "device": device,
            "coordinate_frame": "saved_images_from_ultralytics_results_orig_img",
            "box_source": "ultralytics_results_boxes_xyxy",
        },
        "classes": names,
        "images": images,
        "annotations": annotations,
    }
    predictions_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return PredictionBundleResult(
        output_dir=output_dir,
        predictions_path=predictions_path,
        total_images=len(images),
        total_boxes=len(annotations),
    )


def summarize_prediction_bundle(result: PredictionBundleResult) -> str:
    lines = [
        f"Prediction viewer bundle: {result.output_dir}",
        f"Predictions: {result.predictions_path}",
        f"Images: {result.total_images}",
        f"Boxes: {result.total_boxes}",
    ]
    if result.uploaded_dir:
        lines.append(f"Uploaded bundle: {result.uploaded_dir}")
    return "\n".join(lines)


def run_prediction_bundle_viewer(args: Any) -> None:
    bundle = PredictionBundle(Path(args.bundle))
    app = _create_app(bundle, default_confidence=float(args.conf))
    url = f"http://{args.host}:{args.port}"
    print(f"Prediction bundle viewer: {url}")
    print(f"Bundle: {bundle.bundle_dir}")
    print(f"Images: {len(bundle.images)}")
    print(f"Boxes: {len(bundle.annotations)}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False)


class PredictionBundle:
    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir.expanduser().resolve()
        self.predictions_path = self.bundle_dir / "predictions_viewer.json"
        self.images_dir = self.bundle_dir / "images"
        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Prediction bundle JSON not found: {self.predictions_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Prediction bundle images directory not found: {self.images_dir}")

        data = json.loads(self.predictions_path.read_text(encoding="utf-8"))
        self.classes = [str(name) for name in data.get("classes", [])]
        self.images = [image for image in data.get("images", []) if isinstance(image, Mapping)]
        self.annotations = [ann for ann in data.get("annotations", []) if isinstance(ann, Mapping)]
        self.annotations_by_image_id: dict[int, list[Mapping[str, Any]]] = {}
        for annotation in self.annotations:
            try:
                image_id = int(annotation["image_id"])
            except (KeyError, TypeError, ValueError):
                continue
            self.annotations_by_image_id.setdefault(image_id, []).append(annotation)

    def list_images(self, confidence: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for image in self.images:
            image_id = int(image["id"])
            boxes = self.annotations_by_image_id.get(image_id, [])
            visible = [box for box in boxes if _annotation_confidence(box) >= confidence]
            rows.append(
                {
                    "id": image_id,
                    "filename": image["display_filename"],
                    "source_filename": image.get("source_filename", image["display_filename"]),
                    "width": image["width"],
                    "height": image["height"],
                    "total_boxes": len(boxes),
                    "visible_boxes": len(visible),
                    "max_score": max((_annotation_confidence(box) for box in boxes), default=None),
                }
            )
        return rows

    def boxes_for(self, image_id: int, confidence: float) -> list[dict[str, Any]]:
        boxes = [
            _viewer_box(annotation)
            for annotation in self.annotations_by_image_id.get(image_id, [])
            if _annotation_confidence(annotation) >= confidence
        ]
        return sorted(boxes, key=lambda box: float(box["confidence"]), reverse=True)

    def image_bytes(self, filename: str) -> bytes:
        path = self.images_dir / Path(filename).name
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_bytes()

    def stats(self, confidence: float) -> dict[str, Any]:
        visible = sum(1 for annotation in self.annotations if _annotation_confidence(annotation) >= confidence)
        return {
            "total_images": len(self.images),
            "total_predictions": len(self.annotations),
            "visible_predictions": visible,
        }


def _create_app(bundle: PredictionBundle, default_confidence: float):
    from flask import Flask, abort, jsonify, request, send_file

    app = Flask(__name__)

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/")
    def index():
        return HTML_TEMPLATE

    @app.route("/api/config")
    def config():
        return jsonify({"default_confidence": default_confidence, "classes": bundle.classes})

    @app.route("/api/images")
    def images():
        return jsonify(bundle.list_images(_request_confidence(request.args, default_confidence)))

    @app.route("/api/boxes/<int:image_id>")
    def boxes(image_id: int):
        return jsonify(bundle.boxes_for(image_id, _request_confidence(request.args, default_confidence)))

    @app.route("/api/image/<path:filename>")
    def image(filename: str):
        try:
            image_bytes = bundle.image_bytes(filename)
        except FileNotFoundError:
            abort(404)
        return send_file(BytesIO(image_bytes), mimetype="image/jpeg", download_name=filename)

    @app.route("/api/stats")
    def stats():
        return jsonify(bundle.stats(_request_confidence(request.args, default_confidence)))

    return app


def _request_confidence(args: Any, default: float) -> float:
    try:
        confidence = float(args.get("conf", default))
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _encode_result_image(image: Any) -> tuple[bytes, int, int]:
    if image is None:
        raise RuntimeError("Ultralytics result did not include orig_img.")
    height, width = int(image.shape[0]), int(image.shape[1])
    try:
        import cv2

        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("cv2.imencode failed.")
        return encoded.tobytes(), width, height
    except ImportError:
        from PIL import Image

        rgb = image[:, :, ::-1] if getattr(image, "ndim", 0) == 3 and image.shape[2] == 3 else image
        output = BytesIO()
        Image.fromarray(rgb).convert("RGB").save(output, format="JPEG", quality=95)
        return output.getvalue(), width, height


def _boxes_from_result(result: Any, names: Sequence[str]) -> list[dict[str, Any]]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return []
    xyxy_rows = _to_plain_list(getattr(boxes_obj, "xyxy", []))
    conf_rows = _to_plain_list(getattr(boxes_obj, "conf", []))
    cls_rows = _to_plain_list(getattr(boxes_obj, "cls", []))

    boxes: list[dict[str, Any]] = []
    for index, xyxy in enumerate(xyxy_rows):
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        confidence = float(conf_rows[index]) if index < len(conf_rows) else 0.0
        class_id = int(float(cls_rows[index])) if index < len(cls_rows) else 0
        boxes.append(
            {
                "class_id": class_id,
                "label": _label_for_class(class_id, names),
                "confidence": confidence,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "xyxy": [x1, y1, x2, y2],
            }
        )
    boxes.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return boxes


def _viewer_box(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": annotation.get("id"),
        "bbox": annotation.get("bbox", [0, 0, 0, 0]),
        "xyxy": annotation.get("xyxy"),
        "label": annotation.get("label", f"class_{annotation.get('class_id', 0)}"),
        "class_id": annotation.get("class_id", 0),
        "confidence": _annotation_confidence(annotation),
    }


def _annotation_confidence(annotation: Mapping[str, Any]) -> float:
    try:
        return float(annotation.get("confidence", annotation.get("score", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _to_plain_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=lambda key: int(key))
        except (TypeError, ValueError):
            keys = sorted(names)
        return [str(names[key]) for key in keys]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _label_for_class(class_id: int, names: Sequence[str]) -> str:
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _display_filename(image_path: Path) -> str:
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        return image_path.name
    return f"{image_path.stem}.jpg"


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO Prediction Bundle</title>
<style>
:root {
  --bg-0:#0b0f15; --bg-1:#111820; --bg-2:#1a2230; --border:#2a3a4e;
  --text-1:#e0e8f0; --text-2:#91a0b3; --text-3:#647386;
  --blue:#4f8cff; --cyan:#00c7d9; --green:#00d48a; --mono:Menlo,Consolas,monospace;
}
* { box-sizing:border-box; }
body { margin:0; height:100vh; overflow:hidden; background:var(--bg-0); color:var(--text-1); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.app { display:grid; grid-template-columns:300px 1fr 320px; grid-template-rows:56px 1fr 42px; height:100vh; }
.header { grid-column:1/-1; display:flex; align-items:center; gap:16px; padding:0 16px; background:var(--bg-1); border-bottom:1px solid var(--border); }
.logo { font-family:var(--mono); font-size:13px; font-weight:700; color:var(--blue); }
.stats { display:flex; gap:12px; font-family:var(--mono); font-size:11px; color:var(--text-2); }
.spacer { flex:1; }
.conf { display:flex; align-items:center; gap:10px; width:min(520px, 42vw); }
.conf label { font-family:var(--mono); font-size:11px; color:var(--text-2); white-space:nowrap; }
.conf input { width:100%; accent-color:var(--blue); }
.conf span { font-family:var(--mono); font-size:12px; width:42px; text-align:right; }
.left,.right { background:var(--bg-1); overflow-y:auto; }
.left { border-right:1px solid var(--border); }
.right { border-left:1px solid var(--border); }
.ptitle { font-family:var(--mono); font-size:10px; font-weight:700; color:var(--text-3); text-transform:uppercase; letter-spacing:1.2px; padding:14px 14px 8px; }
.iitem { display:flex; gap:9px; align-items:center; padding:9px 14px; border-left:3px solid transparent; cursor:pointer; }
.iitem:hover { background:var(--bg-2); }
.iitem.active { background:rgba(79,140,255,.12); border-left-color:var(--blue); }
.iname { font-family:var(--mono); font-size:10px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.imeta { font-family:var(--mono); font-size:9px; color:var(--text-3); margin-top:2px; }
.badge { margin-left:auto; font-family:var(--mono); font-size:10px; color:var(--text-2); }
.canvas-wrap { position:relative; overflow:hidden; background:#05080c; }
canvas { display:none; cursor:default; }
.empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--text-3); font-family:var(--mono); font-size:13px; text-align:center; padding:24px; }
.zoom { position:absolute; bottom:12px; left:12px; background:rgba(17,24,32,.9); border:1px solid var(--border); color:var(--text-2); font-family:var(--mono); font-size:11px; padding:6px 8px; border-radius:4px; display:none; gap:6px; align-items:center; }
.zoom button { background:var(--bg-2); border:1px solid var(--border); color:var(--text-2); height:22px; cursor:pointer; }
.bitem { padding:9px 12px; border-bottom:1px solid var(--border); border-left:3px solid transparent; cursor:pointer; }
.bitem:hover { background:var(--bg-2); }
.bitem.hl { background:rgba(79,140,255,.12); border-left-color:var(--blue); }
.bh { display:flex; justify-content:space-between; align-items:center; font-family:var(--mono); font-size:10px; color:var(--text-3); }
.lbl { margin-top:3px; font-family:var(--mono); font-size:11px; color:var(--text-1); overflow:hidden; text-overflow:ellipsis; }
.coords { margin-top:3px; font-family:var(--mono); font-size:9px; color:var(--text-3); }
.footer { grid-column:1/-1; display:flex; align-items:center; gap:18px; padding:0 16px; background:var(--bg-1); border-top:1px solid var(--border); color:var(--text-3); font-family:var(--mono); font-size:10px; }
kbd { border:1px solid var(--border); background:var(--bg-2); border-radius:3px; padding:1px 5px; color:var(--text-2); }
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="logo">YOLO PREDICTIONS</div>
    <div class="stats"><span id="sImg">0 images</span><span id="sVis">0 visible</span><span id="sTot">0 total</span></div>
    <div class="spacer"></div>
    <div class="conf"><label for="confInput">confidence</label><input id="confInput" type="range" min="0" max="1" step="0.01" value="0.25"><span id="confValue">0.25</span></div>
  </div>
  <div class="left"><div class="ptitle">Images</div><div id="ilist"></div></div>
  <div class="canvas-wrap" id="cw"><div class="empty" id="empty">Select an image</div><canvas id="cv"></canvas><div class="zoom" id="zoom"><span id="zoomLbl">100%</span><button onclick="zoomBy(1/1.25)">-</button><button onclick="zoomReset()">reset</button><button onclick="zoomBy(1.25)">+</button></div></div>
  <div class="right"><div class="ptitle">Detections</div><div id="blist"></div></div>
  <div class="footer"><span><kbd>←</kbd><kbd>→</kbd> image</span><span><kbd>↑</kbd><kbd>↓</kbd> detection</span><span>drag to pan</span><span>scroll to zoom</span></div>
</div>
<script>
let imgs=[], ci=-1, cb=-1, conf=0.25, boxCache={}, imgCache={}, scale=1, baseScale=1, pan={x:0,y:0};
let panning=false, panStart=null;
const cv=document.getElementById('cv'), cx=cv.getContext('2d'), cw=document.getElementById('cw');
const confInput=document.getElementById('confInput'), confValue=document.getElementById('confValue');
async function fetchJSON(url){const r=await fetch(url,{cache:'no-store'}); if(!r.ok) throw new Error(await r.text()); return await r.json();}
function confUrl(url){return `${url}${url.includes('?')?'&':'?'}conf=${conf}`;}
async function init(){const cfg=await fetchJSON('/api/config'); conf=Number(cfg.default_confidence??0.25); setConfLabels(); await refreshImages(); await updateStats(); if(imgs.length) selectImg(0);}
function setConfLabels(){const label=conf.toFixed(2); confInput.value=label; confValue.textContent=label;}
async function refreshImages(){const current=imgs[ci]?imgs[ci].id:null; imgs=await fetchJSON(confUrl('/api/images')); if(current){const idx=imgs.findIndex(x=>x.id===current); ci=idx>=0?idx:Math.min(ci,imgs.length-1);} renderList();}
function renderList(){document.getElementById('ilist').innerHTML=imgs.map((m,i)=>`<div class="iitem ${i===ci?'active':''}" onclick="selectImg(${i})"><div style="min-width:0;flex:1"><div class="iname">${escapeHtml(m.source_filename)}</div><div class="imeta">${m.visible_boxes}/${m.total_boxes} visible</div></div><span class="badge">${m.visible_boxes}</span></div>`).join('');}
async function selectImg(idx){if(idx<0||idx>=imgs.length)return; ci=idx; cb=-1; renderList(); const m=imgs[ci]; document.getElementById('empty').style.display='none'; cv.style.display='block'; document.getElementById('zoom').style.display='flex'; boxCache[m.id]=await fetchJSON(confUrl(`/api/boxes/${m.id}`)); await loadImage(m.filename); resetView(); renderBoxes();}
async function loadImage(filename){if(imgCache[filename])return imgCache[filename]; return new Promise((resolve,reject)=>{const el=new Image(); el.onload=()=>{imgCache[filename]=el; resolve(el);}; el.onerror=reject; el.src=`/api/image/${encodeURIComponent(filename)}?t=${Date.now()}`;});}
function currentImage(){return imgs[ci]?imgCache[imgs[ci].filename]:null;}
function currentBoxes(){return imgs[ci]?(boxCache[imgs[ci].id]||[]):[];}
function resetView(){const el=currentImage(); if(!el)return; const aw=cw.clientWidth, ah=cw.clientHeight; cv.width=aw; cv.height=ah; baseScale=Math.min(aw/el.naturalWidth,ah/el.naturalHeight)*0.95; scale=baseScale; pan={x:(aw-el.naturalWidth*scale)/2,y:(ah-el.naturalHeight*scale)/2}; updateZoom(); paint();}
function paint(){const el=currentImage(); if(!el)return; cx.clearRect(0,0,cv.width,cv.height); cx.save(); cx.translate(pan.x,pan.y); cx.scale(scale,scale); cx.drawImage(el,0,0); currentBoxes().forEach((b,i)=>{const [x,y,w,h]=b.bbox; const hl=i===cb, col=hl?'#00d48a':'#00c7d9'; cx.strokeStyle=col; cx.lineWidth=(hl?3:1.5)/scale; cx.strokeRect(x,y,w,h); cx.fillStyle=hl?'rgba(0,212,138,.12)':'rgba(0,199,217,.08)'; cx.fillRect(x,y,w,h); const lbl=`${b.label} ${Number(b.confidence).toFixed(3)}`; cx.font=`${10/scale}px Menlo,Consolas,monospace`; const tw=cx.measureText(lbl).width; const ly=Math.max(0,y-14/scale); cx.fillStyle=col; cx.fillRect(x,ly,tw+5/scale,14/scale); cx.fillStyle='#fff'; cx.fillText(lbl,x+2/scale,ly+10/scale);}); cx.restore();}
function renderBoxes(){const boxes=currentBoxes(); document.getElementById('blist').innerHTML=boxes.length?boxes.map((b,i)=>`<div class="bitem ${i===cb?'hl':''}" onclick="selectBox(${i})" id="b${i}"><div class="bh"><span>#${escapeHtml(b.id)}</span><span>${Number(b.confidence).toFixed(3)}</span></div><div class="lbl">${escapeHtml(b.label)}</div><div class="coords">${b.bbox.map(v=>Number(v).toFixed(1)).join(', ')}</div></div>`).join(''):'<div class="bitem"><div class="lbl">No detections at this confidence</div></div>';}
function selectBox(i){cb=i; renderBoxes(); paint(); const el=document.getElementById('b'+i); if(el)el.scrollIntoView({block:'nearest'});}
async function updateStats(){const s=await fetchJSON(confUrl('/api/stats')); document.getElementById('sImg').textContent=`${s.total_images} images`; document.getElementById('sVis').textContent=`${s.visible_predictions} visible`; document.getElementById('sTot').textContent=`${s.total_predictions} total`;}
function zoomBy(factor){const el=currentImage(); if(!el)return; const mx=cv.width/2,my=cv.height/2,ix=(mx-pan.x)/scale,iy=(my-pan.y)/scale; scale=Math.max(baseScale*.25,Math.min(scale*factor,baseScale*20)); pan.x=mx-ix*scale; pan.y=my-iy*scale; updateZoom(); paint();}
function zoomReset(){resetView();}
function updateZoom(){document.getElementById('zoomLbl').textContent=`${Math.round(scale/baseScale*100)}%`;}
function canvasToImage(clientX,clientY){const r=cv.getBoundingClientRect(); return{x:(clientX-r.left-pan.x)/scale,y:(clientY-r.top-pan.y)/scale};}
function hitTest(ix,iy){const boxes=currentBoxes(); for(let i=boxes.length-1;i>=0;i--){const [x,y,w,h]=boxes[i].bbox; if(ix>=x&&ix<=x+w&&iy>=y&&iy<=y+h)return i;} return -1;}
cv.addEventListener('mousedown',e=>{panStart={x:e.clientX,y:e.clientY,panX:pan.x,panY:pan.y,moved:false}; panning=true; cv.style.cursor='grabbing';});
cv.addEventListener('mousemove',e=>{if(!panning||!panStart)return; const dx=e.clientX-panStart.x,dy=e.clientY-panStart.y; if(Math.abs(dx)>3||Math.abs(dy)>3)panStart.moved=true; pan.x=panStart.panX+dx; pan.y=panStart.panY+dy; paint();});
cv.addEventListener('mouseup',e=>{if(!panning)return; panning=false; cv.style.cursor='default'; if(panStart&&!panStart.moved){const pt=canvasToImage(e.clientX,e.clientY),hit=hitTest(pt.x,pt.y); if(hit>=0)selectBox(hit);} panStart=null;});
cv.addEventListener('mouseleave',()=>{panning=false; cv.style.cursor='default'; panStart=null;});
cv.addEventListener('wheel',e=>{e.preventDefault(); const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,ix=(mx-pan.x)/scale,iy=(my-pan.y)/scale; scale=Math.max(baseScale*.25,Math.min(scale*(e.deltaY<0?1.15:1/1.15),baseScale*20)); pan.x=mx-ix*scale; pan.y=my-iy*scale; updateZoom(); paint();},{passive:false});
confInput.addEventListener('input',async()=>{conf=Number(confInput.value); setConfLabels(); await refreshImages(); if(imgs[ci]){boxCache[imgs[ci].id]=await fetchJSON(confUrl(`/api/boxes/${imgs[ci].id}`));} renderBoxes(); paint(); await updateStats();});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return; const boxes=currentBoxes(); if(e.key==='ArrowRight'&&ci<imgs.length-1){e.preventDefault(); selectImg(ci+1);} if(e.key==='ArrowLeft'&&ci>0){e.preventDefault(); selectImg(ci-1);} if(e.key==='ArrowDown'&&cb<boxes.length-1){e.preventDefault(); selectBox(cb+1);} if(e.key==='ArrowUp'&&cb>0){e.preventDefault(); selectBox(cb-1);}});
window.addEventListener('resize',resetView);
function escapeHtml(value){return String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');}
init().catch(err=>{document.getElementById('empty').textContent=err.message;});
</script>
</body>
</html>"""
