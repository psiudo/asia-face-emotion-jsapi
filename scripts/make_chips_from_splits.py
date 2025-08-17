# [FILEPATH] scripts/make_chips_from_splits.py
# -*- coding: utf-8 -*-
"""
Splits CSV (train_*.csv, val_*.csv, test_*.csv) 를 읽어
원본 이미지에서 얼굴칩(정사각, 리사이즈)을 추출하고,
칩 경로와 라벨을 담은 CSV를 생성한다.

실행 예:
  python scripts/make_chips_from_splits.py --config data.yaml
옵션:
  --size 160         # data.yaml의 chips.size를 덮어쓰기
  --margin 0.15      # data.yaml의 chips.margin_ratio 덮어쓰기
  --workers 8        # 병렬 저장(스레드)
"""

import os, csv, math, argparse, yaml, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def normpath_posix(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_csv(path: str) -> List[Dict[str,str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            rows.append(row)
    return rows

def write_csv(path: str, rows: List[Dict[str,Any]]):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Windows 한글경로 대응: np.fromfile + imdecode"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def imwrite_unicode(path: str, img: np.ndarray, ext: str = ".jpg", params=None) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if params is None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    try:
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False

def clamp_box(x1,y1,x2,y2,w,h):
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def expand_box(x1,y1,x2,y2,w,h, margin_ratio=0.15):
    bw = x2 - x1
    bh = y2 - y1
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    return clamp_box(x1-mx, y1-my, x2+mx, y2+my, w, h)

def crop_square_chip(img, x1,y1,x2,y2, out_size=160):
    """사각형 → 정사각 패딩 → 리사이즈"""
    h, w = img.shape[:2]
    x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,w,h)
    roi = img[y1:y2, x1:x2]  # HxW

    hh, ww = roi.shape[:2]
    side = max(hh, ww)
    # 중앙 정사각 패딩
    top = (side - hh)//2
    bottom = side - hh - top
    left = (side - ww)//2
    right = side - ww - left
    roi_sq = cv2.copyMakeBorder(roi, top,bottom,left,right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    chip = cv2.resize(roi_sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return chip

def parse_version_from_path(p: str) -> str:
    # 경로에 TRAIN_01 / TRAIN_04 가 들어있으면 버전 추출
    p = p.upper()
    if "TRAIN_01" in p: return "TRAIN_01"
    if "TRAIN_04" in p: return "TRAIN_04"
    return "UNKNOWN"

def process_one(row, root_dir, chips_cfg, mapping_tag, split_name) -> Optional[Dict[str,Any]]:
    """
    row: split CSV 한 행. 요구 컬럼:
         - path (원본 상대경로)
         - label_en (라벨, M1/M2에 따라 다름)
         - minX,minY,maxX,maxY (정수)
         - orig_kor (원본 한글 라벨)
    """
    rel = row["path"]
    full = os.path.join(root_dir, rel)
    label_en = row["label_en"]
    x1 = int(row["minX"]); y1 = int(row["minY"]); x2 = int(row["maxX"]); y2 = int(row["maxY"])

    img = imread_unicode(full)
    if img is None:
        return None
    h,w = img.shape[:2]
    x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,w,h)
    x1,y1,x2,y2 = expand_box(x1,y1,x2,y2,w,h, margin_ratio=chips_cfg["margin_ratio"])

    chip = crop_square_chip(img, x1,y1,x2,y2, out_size=chips_cfg["size"])

    # 출력 경로: data_shared/cropped_faces_160/M1/train/<label_en>/<basename>.jpg
    basename = os.path.basename(rel)
    stem, _ = os.path.splitext(basename)
    out_dir = os.path.join(chips_cfg["out_dir_images"], mapping_tag, split_name, label_en)
    out_path = os.path.join(out_dir, f"{stem}.jpg")
    ok = imwrite_unicode(out_path, chip, ext=".jpg")
    if not ok:
        return None

    rec = {
        "path": normpath_posix(os.path.relpath(out_path, root_dir)),
        "label_en": label_en,
        "orig_kor": row.get("orig_kor",""),
        "split": split_name,
        "mapping": mapping_tag,
        "version": parse_version_from_path(rel),
        "src": normpath_posix(rel),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2
    }
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="data.yaml 경로")
    ap.add_argument("--size", type=int, default=None, help="칩 한 변 크기(override)")
    ap.add_argument("--margin", type=float, default=None, help="박스 여유 비율(override)")
    ap.add_argument("--workers", type=int, default=0, help="저장 병렬 스레드 (0=단일)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root_dir = os.path.abspath(cfg.get("project_root") or ".")
    os.chdir(root_dir)

    chips_cfg = dict(cfg.get("chips") or {})
    if args.size is not None:   chips_cfg["size"] = args.size
    if args.margin is not None: chips_cfg["margin_ratio"] = args.margin

    # 입력 split CSV들이 있는 위치
    splits_dir = os.path.normpath(cfg["outputs"]["splits_dir"])

    # 출력
    out_img_root = os.path.normpath(chips_cfg["out_dir_images"])
    out_csv_root = os.path.normpath(chips_cfg["out_dir_csv"])
    os.makedirs(out_img_root, exist_ok=True)
    os.makedirs(out_csv_root, exist_ok=True)

    mappings: List[str] = list(chips_cfg.get("mappings") or ["M1","M2"])
    splits:   List[str] = list(chips_cfg.get("splits") or ["train","val","test"])

    summary = []

    for mtag in mappings:
        for sp in splits:
            in_csv = os.path.join(splits_dir, f"{sp}_{mtag}.csv")
            if not os.path.isfile(in_csv):
                print(f"[skip] not found: {in_csv}")
                continue

            rows = read_csv(in_csv)
            if not rows:
                print(f"[warn] empty CSV: {in_csv}")
                continue

            # 필수 컬럼 체크
            need = {"path","label_en","minX","minY","maxX","maxY"}
            if not need.issubset(set(rows[0].keys())):
                raise RuntimeError(f"{in_csv}: required columns missing {need - set(rows[0].keys())}")

            out_rows = []
            errors = 0

            if args.workers and args.workers > 0:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(process_one, r, root_dir, chips_cfg, mtag, sp) for r in rows]
                    for i,f in enumerate(as_completed(futs), 1):
                        try:
                            rec = f.result()
                            if rec: out_rows.append(rec)
                            else: errors += 1
                        except Exception:
                            errors += 1
                            traceback.print_exc()
                        if i % 500 == 0:
                            print(f"[{mtag}/{sp}] processed {i}/{len(rows)} ...")
            else:
                for i, r in enumerate(rows, 1):
                    try:
                        rec = process_one(r, root_dir, chips_cfg, mtag, sp)
                        if rec: out_rows.append(rec)
                        else: errors += 1
                    except Exception:
                        errors += 1
                        traceback.print_exc()
                    if i % 500 == 0:
                        print(f"[{mtag}/{sp}] processed {i}/{len(rows)} ...")

            out_csv = os.path.join(out_csv_root, f"{sp}_{mtag}.csv")
            write_csv(out_csv, out_rows)

            print(f"[write] chips CSV: {out_csv}  rows={len(out_rows):,}  errors={errors:,}")

            # 요약
            ver01 = sum(1 for r in out_rows if r["version"]=="TRAIN_01")
            ver04 = sum(1 for r in out_rows if r["version"]=="TRAIN_04")
            summary.append({"mapping": mtag, "split": sp, "rows": len(out_rows), "ver01": ver01, "ver04": ver04})

    # 간단 요약 출력
    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"{s['mapping']:>2} {s['split']:<5} rows={s['rows']:,}  TRAIN_01={s['ver01']:,}  TRAIN_04={s['ver04']:,}")

if __name__ == "__main__":
    main()
