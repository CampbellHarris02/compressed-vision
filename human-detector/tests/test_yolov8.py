#!/usr/bin/env python3
"""
YOLOv8 sanity-check on 10 images (5 from folder 0, 5 from folder 1).

• Only keeps 'person' boxes (class-id 0) with confidence ≥ 0.40
• Saves annotated frames to tests/yolo_out/
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

# ───────── CONFIG ──────────────────────────────────────────────────────
MODEL_WEIGHTS  = "yolov8n.pt"           # nano is fast; change if you want
DATA_ROOT      = Path("human-dataset")
OUT_DIR        = Path("tests/yolo_out")
CONF_THRESH    = 0.40                   # ≥ 40 % confidence
IMAGES_PER_SUB = 5                      # take first 5 from each subfolder

# ───────── Gather 5 + 5 images ────────────────────────────────────────
img_paths = []
for sub in ("0", "1"):
    sub_imgs = sorted((DATA_ROOT / sub).glob("*.png"))[:IMAGES_PER_SUB]
    img_paths.extend(sub_imgs)

if len(img_paths) < 10:
    raise SystemExit("Not enough images found; check DATA_ROOT.")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── Load YOLO model ─────────────────────────────────────────────
print(f"🔍 Loading {MODEL_WEIGHTS}")
model = YOLO(MODEL_WEIGHTS)

# ───────── Process images ─────────────────────────────────────────────
n_person_hits = 0
for p in img_paths:
    res = model(str(p), conf=CONF_THRESH, verbose=False)[0]

    # keep only person boxes ≥ threshold
    keep = [i for i, b in enumerate(res.boxes.data)
            if int(b[5]) == 0 and float(b[4]) >= CONF_THRESH]

    filt = res.boxes.data[keep]
    res.update(boxes=filt)

    if len(keep):
        n_person_hits += 1

    ann = res.plot()                      # numpy BGR
    cv2.imwrite(str(OUT_DIR / p.name), ann)
    print(f"✓ {p.name:25s}  persons={len(keep)}")


# ───────── Summary ────────────────────────────────────────────────────
print("\n──────── Summary ────────")
print(f"Images processed : {len(img_paths)}")
print(f"Images w/ person : {n_person_hits}")
print(f"Outputs saved to : {OUT_DIR.resolve()}")
