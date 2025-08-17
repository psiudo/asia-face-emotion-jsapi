# [FILEPATH] scripts/build_splits_from_kaist.py
# -*- coding: utf-8 -*-
"""
KAIST 원천 이미지(01/04 등 복수 버전) + 라벨 JSON을 결합하여
- master_train.csv / master_val.csv / master_test.csv  (orig_kor 유지)
- train_M1.csv / val_M1.csv / test_M1.csv
- train_M2.csv / val_M2.csv / test_M2.csv
을 생성하고, 생성 결과를 요약한 split_summary.json을 함께 출력합니다.

실행:
  python scripts/build_splits_from_kaist.py --config data.yaml

요구 라이브러리:
  pip install pyyaml
"""
import os, sys, csv, json, math, argparse, random, yaml, datetime
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

# ----------------------------
# 경로/유틸
# ----------------------------
def normpath_posix(p: str) -> str:
    # CSV에는 슬래시 통일(윈도우/리눅스 호환)
    return os.path.normpath(p).replace("\\", "/")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_csv_rows(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        # 비어있어도 파일은 만들어두고 싶으면 주석 해제
        # open(path, "w", newline="", encoding="utf-8").close()
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ----------------------------
# 파일 인덱싱 (원천 이미지)
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def index_images(roots: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    returns:
      path_by_name: {basename: first_full_path_found}
      dup_paths:    {basename: [all_full_paths_found]}
    """
    path_by_name: Dict[str, str] = {}
    dup_paths: Dict[str, List[str]] = defaultdict(list)

    for r in roots:
        if not os.path.isdir(r):
            continue
        for root, _, files in os.walk(r):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMG_EXTS:
                    full = os.path.join(root, f)
                    dup_paths[f].append(full)
                    if f not in path_by_name:
                        path_by_name[f] = full
    return path_by_name, dup_paths

# ----------------------------
# 라벨/박스 선택 정책
# ----------------------------
KOR_LABELS_DEFAULT = ["분노","슬픔","불안","상처","당황","기쁨","중립"]

def majority_label(item: Dict[str, Any], kor_order: List[str], tie_breaker: List[str]) -> Optional[str]:
    votes = []
    for k in ("annot_A","annot_B","annot_C"):
        v = (item.get(k) or {}).get("faceExp")
        if isinstance(v, str) and v.strip():
            votes.append(v.strip())
    if votes:
        c = Counter(votes)
        top_n = max(c.values())
        winners = [k for k,v in c.items() if v == top_n]
        if len(winners) == 1:
            return winners[0]
        # 동률이면 tie_breaker 순서대로 해소
        # 1) uploader
        if "uploader" in tie_breaker:
            up = item.get("faceExp_uploader")
            if isinstance(up, str) and up.strip() in kor_order:
                return up.strip()
        # 2) annot_C/B/A
        for key in ("annot_C","annot_B","annot_A"):
            if key in tie_breaker:
                v = (item.get(key) or {}).get("faceExp")
                if isinstance(v, str) and v.strip() in winners:
                    return v.strip()
        # 3) kor_order 우선순
        for k in kor_order:
            if k in winners:
                return k
    else:
        up = item.get("faceExp_uploader")
        if isinstance(up, str) and up.strip() in kor_order:
            return up.strip()
    return None

def get_box(item: Dict[str, Any], prefer_label: Optional[str],
            policy: Dict[str, Any]) -> Optional[Tuple[int,int,int,int]]:
    """박스는 annot_A/B/C 중 유효 좌표만 고려."""
    min_area = int(policy.get("min_box_area", 16))
    prefer_matched = bool(policy.get("box_prefer_label_matched", True))
    priority = list(policy.get("box_priority", ["annot_C","annot_B","annot_A"]))

    cand = []
    for src in ("annot_A","annot_B","annot_C"):
        ent = item.get(src) or {}
        b = ent.get("boxes") or {}
        try:
            x1 = float(b.get("minX")); y1 = float(b.get("minY"))
            x2 = float(b.get("maxX")); y2 = float(b.get("maxY"))
            if any(map(lambda x: x is None or (isinstance(x,float) and (math.isinf(x) or math.isnan(x))), [x1,y1,x2,y2])):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            lab = ent.get("faceExp")
            cand.append((src, lab, (x1,y1,x2,y2), area))
        except Exception:
            continue

    if not cand:
        return None

    if prefer_label and prefer_matched:
        cand2 = [(src,lab,box,a) for (src,lab,box,a) in cand if isinstance(lab,str) and lab.strip()==prefer_label]
        if cand2:
            cand = cand2

    def first_by_priority(cands):
        for p in priority:
            for (src, lab, box, a) in cands:
                if src == p:
                    return box
        return cands[0][2]

    x1,y1,x2,y2 = map(int, map(round, first_by_priority(cand)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1,y1,x2,y2)

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="data.yaml 경로")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = os.path.abspath(cfg.get("project_root") or ".")
    os.chdir(root)

    # 경로 설정
    image_roots = [os.path.normpath(p) for p in cfg["raw"]["image_roots"]]
    json_dir = os.path.normpath(cfg["json"]["dir"])
    json_files = cfg["json"].get("files") or []
    out_dir = os.path.normpath(cfg["outputs"]["splits_dir"])
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "split_summary.json")

    # 라벨/매핑/스플릿 설정
    kor_order = list(cfg["labels"].get("kor_order") or KOR_LABELS_DEFAULT)
    map_M1 = dict(cfg["labels"]["map_M1"])
    map_M2 = dict(cfg["labels"]["map_M2"])
    map_K7 = dict(cfg["labels"].get("map_K7", {}))
    
    split_cfg = cfg.get("split") or {}
    SEED = int(split_cfg.get("seed", 20250811))
    TEST_RATIO = float(split_cfg.get("test_ratio", 0.2))
    VAL_RATIO = float(split_cfg.get("val_ratio", 0.1))
    PER_CLASS_RAW = split_cfg.get("per_class_cap_raw", None)
    if PER_CLASS_RAW is not None:
        PER_CLASS_RAW = int(PER_CLASS_RAW)
    policy = cfg.get("selection_policy") or {}
    tie_breaker = policy.get("tie_breaker", ["uploader","annot_C","annot_B","annot_A"])
    rng = random.Random(SEED)

    summary: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "project_root": root,
            "image_roots": image_roots,
            "json_dir": json_dir,
            "json_files": json_files,
            "splits_dir": out_dir,
            "seed": SEED,
            "test_ratio": TEST_RATIO,
            "val_ratio": VAL_RATIO,
            "per_class_cap_raw": PER_CLASS_RAW,
            "selection_policy": policy,
        },
        "index": {},
        "collect": {},
        "splits": {},
        "files_created": [],
    }

    # 1) 원천 이미지 인덱싱
    path_by_name, dup_paths = index_images(image_roots)
    n_dup_keys = sum(1 for v in dup_paths.values() if len(v) > 1)
    summary["index"] = {
        "images_indexed": len(path_by_name),
        "duplicate_name_keys": n_dup_keys,
    }
    print(f"[index] images indexed: {len(path_by_name):,} (duplicate basenames: {n_dup_keys:,})")

    # 2) JSON 로드/수집
    master_rows: List[Dict[str, Any]] = []
    dropped_stats = Counter()
    json_stats = []

    for jf in json_files:
        p = os.path.join(json_dir, jf)
        if not os.path.isfile(p):
            print(f"[warn] JSON not found: {p}")
            json_stats.append({"file": jf, "found": False, "items": 0})
            continue
        data = json.load(open(p, "r", encoding="utf-8"))
        if not isinstance(data, list):
            print(f"[warn] JSON root is not list: {jf}")
            json_stats.append({"file": jf, "found": True, "items": 0, "bad_root": True})
            continue

        kept = 0
        for it in data:
            fn = it.get("filename")
            if not fn or fn not in path_by_name:
                dropped_stats["missing_image"] += 1
                continue

            lab_kor = majority_label(it, kor_order, tie_breaker)
            if not lab_kor or lab_kor not in kor_order:
                dropped_stats["bad_label"] += 1
                continue

            box = get_box(it, lab_kor, policy)
            if not box:
                dropped_stats["no_valid_box"] += 1
                continue

            full = path_by_name[fn]
            rel = normpath_posix(os.path.relpath(full, root))
            x1,y1,x2,y2 = box
            master_rows.append({
                "path": rel,
                "orig_kor": lab_kor,
                "minX": x1, "minY": y1, "maxX": x2, "maxY": y2
            })
            kept += 1

        json_stats.append({"file": jf, "found": True, "items": len(data), "kept": kept})

    summary["collect"] = {
        "master_candidates": len(master_rows),
        "dropped": dict(dropped_stats),
        "json_files": json_stats,
    }
    print(f"[collect] master candidates: {len(master_rows):,}")
    if dropped_stats:
        print(f"[collect] dropped counts: {dict(dropped_stats)}")

    if not master_rows:
        print("[error] no rows collected; check config paths.")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        sys.exit(1)

    # 3) per-class 상한 (원본 라벨 기준)
    by_kor: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
    for r in master_rows:
        by_kor[r["orig_kor"]].append(r)
    if PER_CLASS_RAW is not None:
        for k, lst in by_kor.items():
            if len(lst) > PER_CLASS_RAW:
                by_kor[k] = rng.sample(lst, PER_CLASS_RAW)
        master_rows = [r for v in by_kor.values() for r in v]
        print(f"[cap] after per-class cap: {len(master_rows):,}")

    # 4) stratified split (원본 라벨 기준) → train/test
    train_rows, test_rows = [], []
    for k in kor_order:
        rows = by_kor.get(k, [])
        if not rows:
            continue
        rows2 = rows[:]
        rng.shuffle(rows2)
        n = len(rows2)
        n_test = int(round(n * TEST_RATIO))
        if n > 1:
            n_test = min(max(1, n_test), n-1)
        else:
            n_test = 0
        test_rows += rows2[:n_test]
        train_rows += rows2[n_test:]

    # 5) train → train/val (원본 라벨 기준)
    by_kor_train: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
    for r in train_rows:
        by_kor_train[r["orig_kor"]].append(r)

    train2, val_rows = [], []
    for k, lst in by_kor_train.items():
        lst2 = lst[:]; rng.shuffle(lst2)
        n = len(lst2)
        n_val = int(round(n * VAL_RATIO)) if n > 1 else 0
        if n > 1:
            n_val = min(max(1, n_val), n-1)
        else:
            n_val = 0
        val_rows += lst2[:n_val]
        train2  += lst2[n_val:]

    # master CSV (orig_kor 유지)
    master_train = [{"path":r["path"], "orig_kor":r["orig_kor"], "minX":r["minX"], "minY":r["minY"], "maxX":r["maxX"], "maxY":r["maxY"]} for r in train2]
    master_val   = [{"path":r["path"], "orig_kor":r["orig_kor"], "minX":r["minX"], "minY":r["minY"], "maxX":r["maxX"], "maxY":r["maxY"]} for r in val_rows]
    master_test  = [{"path":r["path"], "orig_kor":r["orig_kor"], "minX":r["minX"], "minY":r["minY"], "maxX":r["maxX"], "maxY":r["maxY"]} for r in test_rows]

    p_master_train = os.path.join(out_dir, "master_train.csv")
    p_master_val   = os.path.join(out_dir, "master_val.csv")
    p_master_test  = os.path.join(out_dir, "master_test.csv")
    write_csv_rows(p_master_train, master_train)
    write_csv_rows(p_master_val,   master_val)
    write_csv_rows(p_master_test,  master_test)

    # 6) M1/M2 매핑 CSV
    def apply_map(rows, M: Dict[str,str]) -> List[Dict[str,Any]]:
        out = []
        for r in rows:
            en = M.get(r["orig_kor"])
            if not en:
                continue
            out.append({
                "path": r["path"],
                "label_en": en,
                "minX": r["minX"], "minY": r["minY"], "maxX": r["maxX"], "maxY": r["maxY"],
                "orig_kor": r["orig_kor"]
            })
        return out

    files_created = [p_master_train, p_master_val, p_master_test]
    summary["splits"]["master"] = {
        "train_rows": len(master_train),
        "val_rows":   len(master_val),
        "test_rows":  len(master_test),
        "label_dist": {
            "train": dict(Counter([r["orig_kor"] for r in master_train])),
            "val":   dict(Counter([r["orig_kor"] for r in master_val])),
            "test":  dict(Counter([r["orig_kor"] for r in master_test])),
        }
    }

    for tag, M in (("M1", map_M1), ("M2", map_M2), ("K7", map_K7 or {k:k for k in kor_order})):

        tr = apply_map(master_train, M)
        va = apply_map(master_val,   M)
        te = apply_map(master_test,  M)
        p_tr = os.path.join(out_dir, f"train_{tag}.csv")
        p_va = os.path.join(out_dir, f"val_{tag}.csv")
        p_te = os.path.join(out_dir, f"test_{tag}.csv")
        write_csv_rows(p_tr, tr)
        write_csv_rows(p_va, va)
        write_csv_rows(p_te, te)
        files_created += [p_tr, p_va, p_te]
        print(f"[write] {tag}: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

        summary["splits"][tag] = {
            "train_rows": len(tr),
            "val_rows":   len(va),
            "test_rows":  len(te),
            "label_dist": {
                "train": dict(Counter([r["label_en"] for r in tr])),
                "val":   dict(Counter([r["label_en"] for r in va])),
                "test":  dict(Counter([r["label_en"] for r in te])),
            }
        }

    # 7) 요약 JSON 저장
    summary["files_created"] = [normpath_posix(p) for p in files_created]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[done] splits written to:", out_dir)
    print("[done] summary written to:", summary_path)

if __name__ == "__main__":
    main()
