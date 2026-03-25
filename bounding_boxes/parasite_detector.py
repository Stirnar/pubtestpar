import cv2
import numpy as np
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BBox:
    x: int; y: int; w: int; h: int
    confidence: float
    method: str

@dataclass
class DetectionResult:
    image_path: str
    image_w: int; image_h: int
    fov_center: Optional[Tuple[int, int]]
    fov_radius: Optional[int]
    boxes: List[BBox] = field(default_factory=list)

class ParasiteDetector:

    def __init__(self, min_object_frac=0.0003, max_object_frac=0.20,
                 nms_iou_threshold=0.3, border_margin=10):
        self.min_object_frac = min_object_frac
        self.max_object_frac = max_object_frac
        self.nms_iou_threshold = nms_iou_threshold
        self.border_margin = border_margin

    def detect(self, image_path: str) -> DetectionResult:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        result = DetectionResult(image_path=image_path, image_w=w, image_h=h,
                                 fov_center=None, fov_radius=None)

        fov_mask, center, radius = self._detect_fov(img)
        result.fov_center = center
        result.fov_radius = radius

        if fov_mask is None:
            fov_mask = np.ones((h, w), dtype=np.uint8) * 255
            fov_area = h * w
        else:
            fov_area = np.pi * radius * radius

        edge_margin = max(self.border_margin, int(radius * 0.06)) if radius else self.border_margin
        if radius and radius > edge_margin * 2:
            shrunk = np.zeros_like(fov_mask)
            cv2.circle(shrunk, center, int(radius - edge_margin), 255, -1)
            fov_mask = shrunk

        mn = int(fov_area * self.min_object_frac)
        mx = int(fov_area * self.max_object_frac)
        norm = self._normalize(img, fov_mask)

        all_boxes = []
        all_boxes += self._detect_stained(norm, fov_mask, mn, mx)
        all_boxes += self._detect_contrast(norm, fov_mask, mn, mx)
        all_boxes += self._detect_edges(norm, fov_mask, mn, mx)
        all_boxes += self._detect_dark(norm, fov_mask, mn, mx)
        all_boxes += self._detect_anomaly(norm, fov_mask, mn, mx)

        if center and radius:
            filtered = []
            for box in all_boxes:
                if box.w > radius * 1.5 or box.h > radius * 1.5:
                    continue
                bm = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(bm, (box.x, box.y), (box.x+box.w, box.y+box.h), 255, -1)
                overlap = cv2.bitwise_and(bm, fov_mask)
                if box.w * box.h > 0 and cv2.countNonZero(overlap) / (box.w * box.h) > 0.85:
                    filtered.append(box)
            all_boxes = filtered

        result.boxes = self._nms(all_boxes)
        return result

    def _detect_fov(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        for tv in [20, 30, 40, 50]:
            _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            (cx, cy), r = cv2.minEnclosingCircle(largest)
            ca = np.pi * r * r
            if ca == 0: continue
            if area / ca > 0.85 and r > min(h,w)*0.2 and r < max(h,w)*0.6:
                c = (int(cx), int(cy))
                mask = np.zeros((h,w), dtype=np.uint8)
                cv2.circle(mask, c, int(r), 255, -1)
                return mask, c, int(r)
        return None, None, None

    def _normalize(self, img, mask):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out[mask == 0] = 0
        return out

    def _detect_stained(self, img, mask, mn, mx):
        boxes = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for lo, hi, method in [
            ((140,30,50),(180,255,255),"pink_stain"), ((0,30,50),(15,255,255),"red_stain"),
            ((120,30,50),(145,255,255),"purple_stain"), ((10,40,30),(25,255,200),"brown_stain"),
            ((100,40,30),(125,255,200),"blue_stain"),
        ]:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.bitwise_and(m, mask)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            boxes += self._to_boxes(m, mn, mx, method, 0.7)
        return boxes

    def _detect_contrast(self, img, mask, mn, mx):
        boxes = []
        gray = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
        bg = cv2.GaussianBlur(gray, (51,51), 0)
        diff = cv2.absdiff(gray, bg); diff[mask==0] = 0
        for tv in [15, 25, 40]:
            _, b = cv2.threshold(diff, tv, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
            boxes += self._to_boxes(b, mn, mx, f"contrast_t{tv}", {15:0.5,25:0.6,40:0.7}[tv])
        return boxes

    def _detect_edges(self, img, mask, mn, mx):
        gray = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
        edges = cv2.Canny(gray, 30, 100); edges[mask==0] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        d = cv2.dilate(edges, k, iterations=3)
        d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, k, iterations=2)
        return self._to_boxes(d, mn, mx, "edge_density", 0.4)

    def _detect_dark(self, img, mask, mn, mx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mv = cv2.mean(gray, mask=mask)[0]
        if mv < 30: return []
        fg = gray.copy(); fg[mask==0] = 255
        _, b = cv2.threshold(fg, int(max(30, mv*0.6)), 255, cv2.THRESH_BINARY_INV)
        b = cv2.bitwise_and(b, mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
        return self._to_boxes(b, mn, mx, "dark_object", 0.5)

    def _detect_anomaly(self, img, mask, mn, mx):
        boxes = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray[mask==0] = 0
        for ks in [31, 61]:
            lm = cv2.blur(gray, (ks,ks))
            ls = np.sqrt(np.maximum(cv2.blur(gray*gray,(ks,ks)) - lm*lm, 0) + 1e-6)
            z = np.abs(gray - lm) / (ls + 1e-6); z[mask==0] = 0
            for zt in [2.0, 3.0]:
                b = (z > zt).astype(np.uint8) * 255
                b = cv2.bitwise_and(b, mask)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
                b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
                boxes += self._to_boxes(b, mn, mx, f"anomaly_k{ks}_z{zt}", 0.4 if zt==2.0 else 0.6)
        return boxes

    def _to_boxes(self, mask, mn, mx, method, conf):
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a < mn or a > mx: continue
            x,y,w,h = cv2.boundingRect(cnt)
            if max(w,h)/(min(w,h)+1e-6) > 8: continue
            px, py = max(5,int(w*0.1)), max(5,int(h*0.1))
            boxes.append(BBox(x=max(0,x-px), y=max(0,y-py), w=w+2*px, h=h+2*py,
                              confidence=conf, method=method))
        return boxes

    def _nms(self, boxes):
        if not boxes: return []
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        kept, sup = [], set()
        for i, bi in enumerate(boxes):
            if i in sup: continue
            kept.append(bi)
            for j in range(i+1, len(boxes)):
                if j not in sup and self._iou(bi, boxes[j]) > self.nms_iou_threshold:
                    sup.add(j)
        return kept

    @staticmethod
    def _iou(a, b):
        x1,y1 = max(a.x,b.x), max(a.y,b.y)
        x2,y2 = min(a.x+a.w,b.x+b.w), min(a.y+a.h,b.y+b.h)
        if x2<=x1 or y2<=y1: return 0.0
        inter = (x2-x1)*(y2-y1)
        return inter / (a.w*a.h + b.w*b.h - inter)

def results_to_coco(results, output_path):
    images, annotations = [], []
    ann_id = 1
    for img_id, r in enumerate(results, 1):
        images.append({"id": img_id, "file_name": os.path.basename(r.image_path),
                       "width": r.image_w, "height": r.image_h})
        for box in r.boxes:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [box.x, box.y, box.w, box.h], "area": box.w*box.h,
                "iscrowd": 0, "score": round(box.confidence, 3),
                "detection_method": box.method})
            ann_id += 1
    coco = {
        "info": {"description": "Parasite detection proposals", "version": "1.0",
                 "generator": "parasite_detector.py"},
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "proposal", "supercategory": "detection"}]
    }
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

def results_to_yolo(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        with open(os.path.join(output_dir, Path(r.image_path).stem + ".txt"), 'w') as f:
            for b in r.boxes:
                f.write(f"0 {(b.x+b.w/2)/r.image_w:.6f} {(b.y+b.h/2)/r.image_h:.6f} "
                        f"{b.w/r.image_w:.6f} {b.h/r.image_h:.6f}\n")

def draw_results(image_path, result, output_path):
    img = cv2.imread(image_path)
    colors = {'pink_stain':(180,105,255),'red_stain':(0,0,255),'purple_stain':(255,0,128),
              'brown_stain':(0,140,255),'blue_stain':(255,100,0),'contrast':(0,255,0),
              'edge_density':(255,255,0),'dark_object':(0,200,200),'anomaly':(0,255,128)}
    for box in result.boxes:
        color = next((c for k,c in colors.items() if k in box.method), (0,255,0))
        cv2.rectangle(img, (box.x,box.y), (box.x+box.w,box.y+box.h), color, 2)
        label = f"{box.method} {box.confidence:.1f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (box.x,box.y-th-6), (box.x+tw+4,box.y), color, -1)
        cv2.putText(img, label, (box.x+2,box.y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    if result.fov_center and result.fov_radius:
        cv2.circle(img, result.fov_center, result.fov_radius, (255,200,0), 1)
    cv2.imwrite(output_path, img)

def run_detection(args):
    """Main execution function for parasite detection."""
    os.makedirs(args.output_dir, exist_ok=True)
    det = ParasiteDetector(args.min_frac, args.max_frac, args.nms_iou)

    exts = {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
    try:
        files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                        if Path(f).suffix.lower() in exts])
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    print(f"Found {len(files)} images in {args.input_dir}")

    coco_path = os.path.join(args.output_dir, "proposals.json")
    existing_coco = None
    already_processed = set()

    if not getattr(args, 'force', False) and os.path.isfile(coco_path):
        try:
            with open(coco_path) as f:
                existing_coco = json.load(f)
            already_processed = {img["file_name"] for img in existing_coco.get("images", [])}
            print(f"  Existing proposals.json has {len(already_processed)} images")
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: could not parse existing {coco_path}, re-processing all")
            existing_coco = None

    new_files = [f for f in files if os.path.basename(f) not in already_processed]
    skipped = len(files) - len(new_files)
    if skipped > 0:
        print(f"  Skipping {skipped} already-processed, {len(new_files)} new to process")
    if not new_files and existing_coco is not None:
        print("  Nothing new to process.")
        return

    if not args.no_vis:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    results_list = []
    for filepath in new_files:
        name = os.path.basename(filepath)
        print(f"  Processing {name}...", end=" ", flush=True)
        result = det.detect(filepath)
        results_list.append(result)
        if not args.no_vis:
            draw_results(filepath, result, os.path.join(vis_dir, f"det_{name}"))
        print(f"found {len(result.boxes)} proposals.")

    if existing_coco is not None and existing_coco.get("images"):
        max_img_id = max(img["id"] for img in existing_coco["images"])
        max_ann_id = max((a["id"] for a in existing_coco["annotations"]), default=0)

        for i, r in enumerate(results_list, 1):
            img_id = max_img_id + i
            existing_coco["images"].append({
                "id": img_id, "file_name": os.path.basename(r.image_path),
                "width": r.image_w, "height": r.image_h
            })
            for box in r.boxes:
                max_ann_id += 1
                existing_coco["annotations"].append({
                    "id": max_ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [box.x, box.y, box.w, box.h], "area": box.w * box.h,
                    "iscrowd": 0, "score": round(box.confidence, 3),
                    "detection_method": box.method
                })

        with open(coco_path, 'w') as f:
            json.dump(existing_coco, f, indent=2)
    else:
        results_to_coco(results_list, coco_path)

    print(f"\n→ Proposals: {coco_path}")

    if args.format in ("yolo","both"):
        yolo_dir = os.path.join(args.output_dir, "yolo_labels")
        results_to_yolo(results_list, yolo_dir)
        print(f"→ YOLO:      {yolo_dir}/")
    if not args.no_vis:
        print(f"→ Viz:       {vis_dir}/")

    new_total = sum(len(r.boxes) for r in results_list)
    all_total = new_total + (len(existing_coco["annotations"]) - new_total if existing_coco else 0)
    print(f"\nNew: {new_total} proposals across {len(new_files)} images")
    if skipped > 0:
        print(f"Total: {all_total} proposals across {len(files)} images (including prior)")
    print(f"\nNext: python3 main.py review --images {args.input_dir} --proposals {coco_path}")
