import os
import json
import uuid
import webbrowser
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response

app = Flask(__name__)

IMAGES_DIR = ""
PROPOSALS = {}       # image_id (int) -> list of proposal dicts from COCO
IMAGE_META = {}      # image_id (int) -> {file_name, width, height}
FNAME_TO_ID = {}     # file_name -> image_id

WORKING = {}

ANNOTATIONS_PATH = None  # where to auto-save


def load_proposals(proposals_path):
    with open(proposals_path) as f:
        coco = json.load(f)

    for img in coco["images"]:
        IMAGE_META[img["id"]] = img
        FNAME_TO_ID[img["file_name"]] = img["id"]

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in PROPOSALS:
            PROPOSALS[img_id] = []
        PROPOSALS[img_id].append(ann)


def init_working_state():
    for img_id, meta in IMAGE_META.items():
        fname = meta["file_name"]
        boxes = []
        for ann in PROPOSALS.get(img_id, []):
            boxes.append({
                "id": str(uuid.uuid4())[:8],
                "bbox": ann["bbox"],
                "label": "Noise",
                "status": "pending",
                "method": ann.get("detection_method", "unknown"),
                "confidence": ann.get("score", 0),
            })
        WORKING[fname] = boxes


def load_existing_annotations(ann_path):
    with open(ann_path) as f:
        data = json.load(f)

    raw_images = data.get("images", {})

    if isinstance(raw_images, list):
        id_to_fname = {img["id"]: img["file_name"] for img in raw_images}
        anns_by_fname = {}
        for ann in data.get("annotations", []):
            fname = id_to_fname.get(ann["image_id"])
            if fname:
                anns_by_fname.setdefault(fname, []).append({
                    "bbox": ann["bbox"],
                    "label": ann.get("label", ""),
                    "status": ann.get("status", "pending"),
                    "method": ann.get("detection_method", ann.get("method", "unknown")),
                })
        saved_by_fname = anns_by_fname
    else:
        saved_by_fname = {}
        for fname, img_data in raw_images.items():
            saved_by_fname[fname] = img_data.get("annotations", [])

    for fname, saved_anns in saved_by_fname.items():
        if fname not in WORKING:
            continue

        working_boxes = WORKING[fname]

        matched_saved = set()
        for wb in working_boxes:
            best_iou, best_idx = 0, -1
            for si, sa in enumerate(saved_anns):
                if si in matched_saved:
                    continue
                iou = _bbox_iou(wb["bbox"], sa["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = si
            if best_iou > 0.5 and best_idx >= 0:
                matched_saved.add(best_idx)
                wb["status"] = saved_anns[best_idx]["status"]
                wb["label"] = saved_anns[best_idx].get("label", "")

        for si, sa in enumerate(saved_anns):
            if si not in matched_saved and sa.get("method", "") == "manual":
                working_boxes.append({
                    "id": str(uuid.uuid4())[:8],
                    "bbox": sa["bbox"],
                    "label": sa.get("label", ""),
                    "status": sa["status"],
                    "method": "manual",
                    "confidence": 1.0,
                })

    print(f"  Loaded existing annotations for {len(saved_by_fname)} images")


def _bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    return inter / (a[2]*a[3] + b[2]*b[3] - inter)


def export_annotations():
    """Export annotations in COCO format (same structure as proposals.json,
    with added status and label fields per annotation)."""
    images, annotations = [], []
    ann_id = 1

    for img_idx, (fname, boxes) in enumerate(sorted(WORKING.items()), 1):
        meta = IMAGE_META.get(FNAME_TO_ID.get(fname, -1), {})
        images.append({
            "id": img_idx, "file_name": fname,
            "width": meta.get("width", 0), "height": meta.get("height", 0),
        })
        for box in boxes:
            annotations.append({
                "id": ann_id, "image_id": img_idx, "category_id": 1,
                "bbox": box["bbox"],
                "area": box["bbox"][2] * box["bbox"][3],
                "iscrowd": 0,
                "score": round(box.get("confidence", 0), 3),
                "detection_method": box["method"],
                "status": box["status"],
                "label": box["label"],
            })
            ann_id += 1

    return {
        "info": {"description": "Parasite annotations", "version": "1.0",
                 "generator": "review_tool.py"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "proposal", "supercategory": "detection"}],
    }


def auto_save():
    if ANNOTATIONS_PATH:
        data = export_annotations()
        with open(ANNOTATIONS_PATH, 'w') as f:
            json.dump(data, f, indent=2)

@app.route("/")
def index():
    return HTML_TEMPLATE

@app.route("/api/images")
def list_images():
    result = []
    for fname in sorted(WORKING.keys()):
        boxes = WORKING[fname]
        meta = IMAGE_META.get(FNAME_TO_ID.get(fname, -1), {})
        pending = sum(1 for b in boxes if b["status"] == "pending")
        result.append({
            "filename": fname,
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "total_boxes": len(boxes),
            "pending": pending,
        })
    return jsonify(result)

@app.route("/api/image/<path:filename>")
def serve_image(filename):
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.isfile(path):
        return "Not found", 404
    return send_file(path)

@app.route("/api/boxes/<path:filename>")
def get_boxes(filename):
    boxes = WORKING.get(filename, [])
    return jsonify(boxes)

@app.route("/api/update_box", methods=["POST"])
def update_box():
    data = request.json
    fname = data["filename"]
    box_id = data["box_id"]
    for box in WORKING.get(fname, []):
        if box["id"] == box_id:
            if "status" in data:
                box["status"] = data["status"]
            if "label" in data:
                box["label"] = data["label"]
            if "bbox" in data:
                box["bbox"] = data["bbox"]
            break
    auto_save()
    return jsonify({"ok": True})

@app.route("/api/add_box", methods=["POST"])
def add_box():
    data = request.json
    fname = data["filename"]
    new_box = {
        "id": str(uuid.uuid4())[:8],
        "bbox": [int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])],
        "label": data.get("label") or "Noise",
        "status": "approved",
        "method": "manual",
        "confidence": 1.0,
    }
    WORKING.setdefault(fname, []).append(new_box)
    auto_save()
    return jsonify(new_box)

@app.route("/api/delete_box", methods=["POST"])
def delete_box():
    data = request.json
    fname = data["filename"]
    box_id = data["box_id"]
    WORKING[fname] = [b for b in WORKING.get(fname, []) if b["id"] != box_id]
    auto_save()
    return jsonify({"ok": True})

@app.route("/api/bulk_update", methods=["POST"])
def bulk_update():
    data = request.json
    status = data["status"]
    fname = data.get("filename")  # None = all images

    targets = [fname] if fname else list(WORKING.keys())
    count = 0
    for fn in targets:
        for box in WORKING.get(fn, []):
            if box["status"] == "pending":
                box["status"] = status
                count += 1
    auto_save()
    return jsonify({"ok": True, "updated": count})

@app.route("/api/stats")
def stats():
    total = approved = rejected = pending = 0
    for boxes in WORKING.values():
        for b in boxes:
            total += 1
            if b["status"] == "approved": approved += 1
            elif b["status"] == "rejected": rejected += 1
            else: pending += 1
    return jsonify({
        "total_images": len(WORKING),
        "total_boxes": total,
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
    })

@app.route("/api/export/annotations")
def export_annotations_route():
    data = export_annotations()
    return Response(
        json.dumps(data, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=annotations.json"}
    )

@app.route("/api/export/coco")
def export_coco():
    images, annotations = [], []
    cat_map, cat_id, ann_id = {}, 1, 1

    for img_idx, (fname, boxes) in enumerate(sorted(WORKING.items()), 1):
        meta = IMAGE_META.get(FNAME_TO_ID.get(fname, -1), {})
        images.append({"id": img_idx, "file_name": fname,
                       "width": meta.get("width",0), "height": meta.get("height",0)})
        for box in boxes:
            if box["status"] != "approved":
                continue
            lbl = box["label"] if box["label"] else "parasite"
            if lbl == "noise":
                continue
            if lbl not in cat_map:
                cat_map[lbl] = cat_id; cat_id += 1
            annotations.append({
                "id": ann_id, "image_id": img_idx, "category_id": cat_map[lbl],
                "bbox": box["bbox"], "area": box["bbox"][2] * box["bbox"][3], "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images, "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in cat_map.items()],
    }
    return Response(
        json.dumps(coco, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=annotations_coco.json"}
    )

@app.route("/api/export/yolo")
def export_yolo():
    label_set = set()
    for boxes in WORKING.values():
        for b in boxes:
            if b["status"] == "approved" and b["label"] and b["label"] != "noise":
                label_set.add(b["label"])
    label_list = sorted(label_set)
    if not label_list:
        label_list = ["parasite"]
    label_to_idx = {l: i for i, l in enumerate(label_list)}

    files = {}
    for fname, boxes in sorted(WORKING.items()):
        meta = IMAGE_META.get(FNAME_TO_ID.get(fname, -1), {})
        w, h = meta.get("width", 1), meta.get("height", 1)
        lines = []
        for box in boxes:
            if box["status"] != "approved":
                continue
            lbl = box["label"] if box["label"] else "parasite"
            if lbl == "noise":
                continue
            cls = label_to_idx.get(lbl, 0)
            bx = box["bbox"]
            lines.append(f"{cls} {(bx[0]+bx[2]/2)/w:.6f} {(bx[1]+bx[3]/2)/h:.6f} "
                         f"{bx[2]/w:.6f} {bx[3]/h:.6f}")
        files[Path(fname).stem + ".txt"] = "\n".join(lines)

    output = {"classes": label_list, "labels": files}
    return Response(
        json.dumps(output, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=yolo_export.json"}
    )

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Parasite Review Tool</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-0: #0b0f15; --bg-1: #111820; --bg-2: #1a2230; --bg-3: #1e2a3a;
    --border: #2a3a4e; --border-hi: #3a5070;
    --text-1: #e0e8f0; --text-2: #8899aa; --text-3: #5a6a7a;
    --green: #00d48a; --green-bg: rgba(0,212,138,0.12);
    --red: #ff4466; --red-bg: rgba(255,68,102,0.12);
    --yellow: #ffaa22; --yellow-bg: rgba(255,170,34,0.12);
    --blue: #4488ff; --blue-bg: rgba(68,136,255,0.10);
    --purple: #aa66ff; --purple-bg: rgba(170,102,255,0.12);
    --mono: 'JetBrains Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:var(--sans); background:var(--bg-0); color:var(--text-1); overflow:hidden; height:100vh; }

.app { display:grid; grid-template-columns:280px 1fr 300px; grid-template-rows:52px 1fr 40px; height:100vh; }

.header { grid-column:1/-1; background:var(--bg-1); border-bottom:1px solid var(--border);
  display:flex; align-items:center; padding:0 16px; gap:14px; z-index:10; }
.logo { font-family:var(--mono); font-weight:700; font-size:13px; color:var(--blue);
  display:flex; align-items:center; gap:6px; }
.logo svg { opacity:0.7; }
.sep { width:1px; height:22px; background:var(--border); }
.stats { display:flex; gap:14px; font-family:var(--mono); font-size:11px; }
.stat { display:flex; align-items:center; gap:5px; }
.stat .d { width:7px; height:7px; border-radius:50%; }
.d.g { background:var(--green); } .d.r { background:var(--red); }
.d.y { background:var(--yellow); } .d.b { background:var(--blue); }
.spacer { flex:1; }
.hdr-btns { display:flex; gap:6px; }

.btn { font-family:var(--mono); font-size:10px; font-weight:500; padding:5px 12px;
  border:1px solid var(--border); background:var(--bg-2); color:var(--text-2);
  border-radius:3px; cursor:pointer; transition:all .12s; text-transform:uppercase;
  letter-spacing:.5px; white-space:nowrap; }
.btn:hover { background:var(--bg-3); color:var(--text-1); border-color:var(--border-hi); }
.btn.primary { background:var(--blue); border-color:var(--blue); color:#fff; }
.btn.primary:hover { opacity:.85; }
.btn.ap { background:var(--green-bg); border-color:var(--green); color:var(--green); }
.btn.rj { background:var(--red-bg); border-color:var(--red); color:var(--red); }
.btn.dr { background:var(--bg-2); border-color:var(--border); color:var(--text-2); }
.btn.dr.active { background:var(--purple); border-color:var(--purple); color:#fff; }

.left { background:var(--bg-1); border-right:1px solid var(--border);
  overflow-y:auto; display:flex; flex-direction:column; }
.ptitle { font-family:var(--mono); font-size:9px; font-weight:600; text-transform:uppercase;
  letter-spacing:1.5px; color:var(--text-3); padding:14px 14px 6px; }
.ilist { flex:1; overflow-y:auto; padding:2px 0; }
.iitem { display:flex; align-items:center; padding:7px 14px; gap:8px; cursor:pointer;
  border-left:3px solid transparent; transition:all .1s; font-size:11px; }
.iitem:hover { background:var(--bg-2); }
.iitem.active { background:var(--blue-bg); border-left-color:var(--blue); }
.iitem .thumb { width:32px; height:32px; border-radius:3px; background:var(--bg-2);
  border:1px solid var(--border); object-fit:cover; flex-shrink:0; }
.iitem .info { flex:1; overflow:hidden; }
.iitem .nm { font-family:var(--mono); font-size:10px; font-weight:500;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.iitem .mt { font-size:9px; color:var(--text-3); font-family:var(--mono); }
.badge { font-family:var(--mono); font-size:9px; padding:2px 5px; border-radius:2px;
  font-weight:600; flex-shrink:0; }
.badge.done { background:var(--green-bg); color:var(--green); }
.badge.wip { background:var(--yellow-bg); color:var(--yellow); }

.canvas-wrap { background:var(--bg-0); position:relative; overflow:hidden;
  display:flex; align-items:center; justify-content:center; }
.canvas-wrap canvas { cursor:default; }
.empty { text-align:center; color:var(--text-3); }
.empty .icon { font-size:40px; margin-bottom:8px; opacity:.3; }
.empty p { font-family:var(--mono); font-size:12px; }

.right { background:var(--bg-1); border-left:1px solid var(--border);
  overflow-y:auto; display:flex; flex-direction:column; }
.box-ctrls { padding:6px 10px; display:flex; gap:4px; border-bottom:1px solid var(--border); }
.blist { flex:1; overflow-y:auto; padding:2px 0; }
.bitem { padding:6px 10px; border-bottom:1px solid var(--border); cursor:pointer;
  transition:all .1s; border-left:3px solid transparent; }
.bitem:hover { background:var(--bg-2); }
.bitem.hl { background:var(--blue-bg); border-left-color:var(--blue); }
.bitem.st-approved { border-left-color:var(--green); }
.bitem.st-rejected { border-left-color:var(--red); opacity:.45; }
.bitem.st-pending { border-left-color:var(--yellow); }
.bitem .bh { display:flex; align-items:center; justify-content:space-between; margin-bottom:3px; }
.bitem .bid { font-family:var(--mono); font-size:9px; color:var(--text-3); }
.bitem .bm { font-family:var(--mono); font-size:9px; padding:1px 4px;
  background:var(--bg-2); border-radius:2px; color:var(--text-2); }
.bitem .bc { font-family:var(--mono); font-size:10px; color:var(--text-2); }
.bitem .ba { display:flex; gap:3px; margin-top:4px; }
.bitem .ba .btn { font-size:9px; padding:2px 6px; }
.linput { width:100%; margin-top:3px; padding:3px 6px; font-family:var(--mono); font-size:10px;
  background:var(--bg-0); border:1px solid var(--border); border-radius:2px;
  color:var(--text-1); outline:none; }
.linput:focus { border-color:var(--blue); }

.footer { grid-column:1/-1; background:var(--bg-1); border-top:1px solid var(--border);
  display:flex; align-items:center; padding:0 16px; font-family:var(--mono);
  font-size:10px; color:var(--text-3); gap:18px; }
kbd { background:var(--bg-2); border:1px solid var(--border); border-radius:2px;
  padding:0 4px; font-family:var(--mono); font-size:9px; color:var(--text-2); }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="logo">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/>
        <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>
        <line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>
      </svg>
      REVIEW TOOL
    </div>
    <div class="sep"></div>
    <div class="stats">
      <div class="stat"><span class="d b"></span><span id="sImg">0 img</span></div>
      <div class="stat"><span class="d g"></span><span id="sOk">0</span></div>
      <div class="stat"><span class="d r"></span><span id="sNo">0</span></div>
      <div class="stat"><span class="d y"></span><span id="sPend">0</span></div>
    </div>
    <div class="spacer"></div>
    <div class="hdr-btns">
      <button class="btn" onclick="bulkAll('approved')">✓ APPROVE ALL</button>
      <button class="btn primary" onclick="location.href='/api/export/annotations'">SAVE JSON</button>
      <button class="btn" onclick="location.href='/api/export/coco'">COCO</button>
      <button class="btn" onclick="location.href='/api/export/yolo'">YOLO</button>
    </div>
  </div>

  <div class="left">
    <div class="ptitle">Images</div>
    <div class="ilist" id="ilist"></div>
  </div>

  <div class="canvas-wrap" id="cw">
    <div class="empty" id="empty"><div class="icon">🔬</div><p>Select an image to review</p></div>
    <canvas id="cv" style="display:none"></canvas>
  </div>

  <div class="right">
    <div class="ptitle">Detections</div>
    <div class="box-ctrls">
      <button class="btn ap" onclick="setAll('approved')">✓ ALL</button>
      <button class="btn rj" onclick="setAll('rejected')">✕ ALL</button>
      <button class="btn dr" onclick="toggleDraw()" id="dbtn">+ DRAW</button>
    </div>
    <div class="blist" id="blist"></div>
  </div>

  <div class="footer">
    <span><kbd>A</kbd> Approve</span>
    <span><kbd>R</kbd> Reject</span>
    <span><kbd>D</kbd> Draw</span>
    <span><kbd>←</kbd><kbd>→</kbd> Images</span>
    <span><kbd>Tab</kbd> Next pending</span>
  </div>
</div>

<script>
let imgs=[], ci=-1, cb=-1, draw=false, ds=null, dragging=false;
let boxes={}, imgCache={}, scale=1, pan={x:0,y:0};
const cv=document.getElementById('cv'), cx=cv.getContext('2d'), cw=document.getElementById('cw');

async function init() {
    const r = await fetch('/api/images');
    imgs = await r.json();
    renderList();
    if (imgs.length > 0) selectImg(0);
    updateStats();
}

function renderList() {
    document.getElementById('ilist').innerHTML = imgs.map((m,i) =>
        `<div class="iitem ${i===ci?'active':''}" onclick="selectImg(${i})">
            <img class="thumb" src="/api/image/${m.filename}" loading="lazy">
            <div class="info"><div class="nm">${m.filename}</div>
            <div class="mt">${m.total_boxes} boxes</div></div>
            ${m.pending===0?'<span class="badge done">done</span>':
              `<span class="badge wip">${m.pending} left</span>`}
        </div>`
    ).join('');
}

async function selectImg(idx) {
    ci=idx; cb=-1;
    const m=imgs[ci];
    document.getElementById('empty').style.display='none';
    cv.style.display='block';

    if (!boxes[m.filename]) {
        const r = await fetch(`/api/boxes/${m.filename}`);
        boxes[m.filename] = await r.json();
    }

    const show = (el) => {
        const aw=cw.clientWidth, ah=cw.clientHeight;
        scale = Math.min(aw/el.naturalWidth, ah/el.naturalHeight)*0.95;
        cv.width=aw; cv.height=ah;
        pan = {x:(aw-el.naturalWidth*scale)/2, y:(ah-el.naturalHeight*scale)/2};
        paint();
    };

    if (imgCache[m.filename]) show(imgCache[m.filename]);
    else {
        const el=new Image();
        el.onload=()=>{ imgCache[m.filename]=el; show(el); };
        el.src=`/api/image/${m.filename}`;
    }
    renderList(); renderBoxes();
}

function paint() {
    if (ci<0) return;
    const m=imgs[ci], el=imgCache[m.filename];
    if (!el) return;
    cx.clearRect(0,0,cv.width,cv.height);
    cx.save(); cx.translate(pan.x,pan.y); cx.scale(scale,scale);
    cx.drawImage(el,0,0);

    const bx = boxes[m.filename]||[];
    bx.forEach((b,i) => {
        const hl = i===cb;
        let col, fill;
        if (b.method==='manual') { col='#aa66ff'; fill='rgba(170,102,255,0.1)'; }
        else if (b.status==='approved') { col='#00d48a'; fill='rgba(0,212,138,0.08)'; }
        else if (b.status==='rejected') { col='#ff4466'; fill='rgba(255,68,102,0.05)'; }
        else { col='#ffaa22'; fill='rgba(255,170,34,0.08)'; }
        if (b.status==='rejected'&&!hl) cx.globalAlpha=0.2;
        cx.fillStyle=fill; cx.fillRect(b.bbox[0],b.bbox[1],b.bbox[2],b.bbox[3]);
        cx.strokeStyle=col; cx.lineWidth=(hl?3:1.5)/scale;
        if (b.status==='rejected'&&!hl) { cx.setLineDash([4/scale,4/scale]); }
        else cx.setLineDash([]);
        cx.strokeRect(b.bbox[0],b.bbox[1],b.bbox[2],b.bbox[3]);
        cx.setLineDash([]);
        const lbl=b.label||b.method;
        cx.font=`${10/scale}px JetBrains Mono`;
        const tw=cx.measureText(lbl).width;
        cx.fillStyle=col; cx.fillRect(b.bbox[0],b.bbox[1]-13/scale,tw+5/scale,13/scale);
        cx.fillStyle='#fff'; cx.fillText(lbl,b.bbox[0]+2/scale,b.bbox[1]-3/scale);
        cx.globalAlpha=1;
    });

    if (draw&&ds&&dragging) {
        cx.strokeStyle='#aa66ff'; cx.lineWidth=2/scale;
        cx.setLineDash([6/scale,3/scale]);
        const x=Math.min(ds.x,ds.cx), y=Math.min(ds.y,ds.cy);
        cx.strokeRect(x,y,Math.abs(ds.cx-ds.x),Math.abs(ds.cy-ds.y));
        cx.setLineDash([]);
    }
    cx.restore();
}

function renderBoxes() {
    const m=imgs[ci];
    if (!m) { document.getElementById('blist').innerHTML=''; return; }
    const bx=boxes[m.filename]||[];
    document.getElementById('blist').innerHTML = bx.map((b,i) =>
        `<div class="bitem st-${b.status} ${i===cb?'hl':''}" onclick="selBox(${i})" id="b${i}">
            <div class="bh">
                <span class="bid">#${b.id}</span>
                <span class="bm">${b.method}</span>
                <span class="bc">${(b.confidence*100).toFixed(0)}%</span>
            </div>
            <input class="linput" placeholder="label…" value="${b.label||''}"
                onclick="event.stopPropagation()"
                onchange="updLabel(${i},this.value)">
            <div class="ba">
                <button class="btn ap" onclick="event.stopPropagation();setBox(${i},'approved')">✓</button>
                <button class="btn rj" onclick="event.stopPropagation();setBox(${i},'rejected')">✕</button>
                <button class="btn" onclick="event.stopPropagation();setBox(${i},'pending')">↺</button>
                <button class="btn" onclick="event.stopPropagation();delBox(${i})" style="margin-left:auto">🗑</button>
            </div>
        </div>`
    ).join('');
}

function selBox(i) { cb=i; renderBoxes(); paint();
    const el=document.getElementById('b'+i);
    if (el) el.scrollIntoView({block:'nearest',behavior:'smooth'}); }

async function setBox(i, st) {
    const m=imgs[ci], bx=boxes[m.filename];
    bx[i].status=st;
    await fetch('/api/update_box',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({filename:m.filename,box_id:bx[i].id,status:st})});
    if (st!=='pending') {
        const nx=bx.findIndex((b,j)=>j>i&&b.status==='pending');
        if (nx>=0) cb=nx;
    }
    refreshImageMeta(m.filename, bx);
    renderBoxes(); renderList(); paint(); updateStats();
}

async function setAll(st) {
    const m=imgs[ci];
    await fetch('/api/bulk_update',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({filename:m.filename,status:st})});
    (boxes[m.filename]||[]).forEach(b=>{ if(b.status==='pending') b.status=st; });
    refreshImageMeta(m.filename, boxes[m.filename]||[]);
    renderBoxes(); renderList(); paint(); updateStats();
}

async function bulkAll(st) {
    await fetch('/api/bulk_update',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({status:st})});
    for (const fn in boxes) boxes[fn].forEach(b=>{ if(b.status==='pending') b.status=st; });
    imgs.forEach(m=>{ m.pending=0; });
    renderBoxes(); renderList(); paint(); updateStats();
}

async function updLabel(i, lbl) {
    const m=imgs[ci], bx=boxes[m.filename];
    bx[i].label=lbl;
    await fetch('/api/update_box',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({filename:m.filename,box_id:bx[i].id,label:lbl})});
    paint();
}

async function delBox(i) {
    const m=imgs[ci], bx=boxes[m.filename];
    await fetch('/api/delete_box',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({filename:m.filename,box_id:bx[i].id})});
    bx.splice(i,1);
    if (cb>=bx.length) cb=bx.length-1;
    refreshImageMeta(m.filename, bx);
    renderBoxes(); renderList(); paint(); updateStats();
}

function refreshImageMeta(fname, bx) {
    const m=imgs.find(x=>x.filename===fname);
    if (m) { m.total_boxes=bx.length; m.pending=bx.filter(b=>b.status==='pending').length; }
}

function toggleDraw() {
    draw=!draw;
    const btn=document.getElementById('dbtn');
    btn.classList.toggle('active',draw);
    cv.style.cursor=draw?'crosshair':'default';
}

cv.addEventListener('mousedown',e=>{
    if (!draw||ci<0) return;
    const r=cv.getBoundingClientRect();
    const ix=(e.clientX-r.left-pan.x)/scale, iy=(e.clientY-r.top-pan.y)/scale;
    ds={x:ix,y:iy,cx:ix,cy:iy}; dragging=true;
});
cv.addEventListener('mousemove',e=>{
    if (!dragging||!ds) return;
    const r=cv.getBoundingClientRect();
    ds.cx=(e.clientX-r.left-pan.x)/scale; ds.cy=(e.clientY-r.top-pan.y)/scale;
    paint();
});
cv.addEventListener('mouseup',async e=>{
    if (!dragging||!ds) return;
    dragging=false;
    const x=Math.round(Math.min(ds.x,ds.cx)), y=Math.round(Math.min(ds.y,ds.cy));
    const w=Math.round(Math.abs(ds.cx-ds.x)), h=Math.round(Math.abs(ds.cy-ds.y));
    ds=null;
    if (w<5||h<5) { paint(); return; }
    const m=imgs[ci];
    const res=await fetch('/api/add_box',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({filename:m.filename,x,y,w,h})});
    const nb=await res.json();
    boxes[m.filename].push(nb);
    cb=boxes[m.filename].length-1;
    refreshImageMeta(m.filename, boxes[m.filename]);
    renderBoxes(); renderList(); paint(); updateStats(); toggleDraw();
});

document.addEventListener('keydown',e=>{
    if (e.target.tagName==='INPUT') return;
    const m=imgs[ci], bx=m?boxes[m.filename]:null;
    switch(e.key) {
        case 'a': case 'A': if(bx&&cb>=0) setBox(cb,'approved'); break;
        case 'r': if(bx&&cb>=0) setBox(cb,'rejected'); break;
        case 'd': case 'D': toggleDraw(); break;
        case 'ArrowRight': e.preventDefault(); if(ci<imgs.length-1) selectImg(ci+1); break;
        case 'ArrowLeft': e.preventDefault(); if(ci>0) selectImg(ci-1); break;
        case 'ArrowDown': e.preventDefault(); if(bx&&cb<bx.length-1) selBox(cb+1); break;
        case 'ArrowUp': e.preventDefault(); if(cb>0) selBox(cb-1); break;
        case 'Tab': e.preventDefault();
            if(bx) { const s=cb+1; for(let i=0;i<bx.length;i++){
                const idx=(s+i)%bx.length;
                if(bx[idx].status==='pending'){selBox(idx);break;}
            }} break;
    }
});

async function updateStats() {
    const r=await fetch('/api/stats'); const s=await r.json();
    document.getElementById('sImg').textContent=s.total_images+' img';
    document.getElementById('sOk').textContent=s.approved+' ✓';
    document.getElementById('sNo').textContent=s.rejected+' ✕';
    document.getElementById('sPend').textContent=s.pending+' ?';
}

window.addEventListener('resize',()=>{
    if(ci>=0){const m=imgs[ci],el=imgCache[m.filename];
    if(el){const aw=cw.clientWidth,ah=cw.clientHeight;
    scale=Math.min(aw/el.naturalWidth,ah/el.naturalHeight)*0.95;
    cv.width=aw;cv.height=ah;
    pan={x:(aw-el.naturalWidth*scale)/2,y:(ah-el.naturalHeight*scale)/2};paint();}}
});

init();
</script>
</body>
</html>"""

def run_review_tool(args):
    global IMAGES_DIR, ANNOTATIONS_PATH

    IMAGES_DIR = os.path.abspath(args.images)
    ANNOTATIONS_PATH = os.path.abspath(args.output)

    print(f"\n{'═'*60}")
    print(f"  🔬  PARASITE REVIEW TOOL")
    print(f"{'═'*60}")
    print(f"  Images:      {IMAGES_DIR}")
    print(f"  Proposals:   {args.proposals}")

    load_proposals(args.proposals)
    init_working_state()

    if args.annotations and os.path.isfile(args.annotations):
        print(f"  Resuming:    {args.annotations}")
        load_existing_annotations(args.annotations)

    total_boxes = sum(len(b) for b in WORKING.values())
    print(f"  Loaded:      {len(WORKING)} images, {total_boxes} boxes")
    print(f"  Auto-save:   {ANNOTATIONS_PATH}")
    print(f"  URL:         http://{args.host}:{args.port}")
    print(f"{'═'*60}\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()

    app.run(host=args.host, port=args.port, debug=False)