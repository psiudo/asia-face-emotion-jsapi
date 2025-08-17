<!-- [FILEPATH] docs/data_prep.md -->
# 데이터 준비 가이드 — data_raw → CSV → 160×160 Crop → 분할(M1/M2/K7)

이 문서는 원천 데이터(**data_raw/**)를 받아 **실험 가능 상태**(CSV + 160×160 크롭)로 만드는 전 과정을 규정합니다.  
최종 산출물은 `data_shared/cropped_faces_csv/` 와 `data_shared/cropped_faces_160/` 입니다.

---

## 0) 디렉토리 규약

```
data_raw/
  └─ aihub_korean_emotion/            # AI Hub 원본(압축 해제본)
     ├─ images/ or frames/ ...        # 정지 이미지 또는 프레임
     └─ annotations/ ...              # 메타/라벨/바운딩박스(JSON/CSV)
data_shared/
  ├─ cropped_faces_csv/
  │   ├─ train_M1.csv  val_M1.csv  test_M1.csv
  │   ├─ train_M2.csv  val_M2.csv  test_M2.csv
  │   └─ train_K7.csv  val_K7.csv  test_K7.csv
  └─ cropped_faces_160/
      ├─ M1/ ...  M2/ ...  K7/ ...   # (선택) 이미지 복사/정리용
```

> **권장 OS 경로 예**: `C:\Users\<YOU>\Desktop\Asia_Face_Emtion_Improve\data_raw\aihub_korean_emotion\`

---

## 1) 요구 패키지

- Python 3.10+
- `pandas`, `numpy`, `Pillow`, `opencv-python`, `tqdm` (선택: `scikit-learn` for split)
- (선택) 탐지/랜드마크를 다시 뽑을 경우: `insightface`/`retinaface`/`mediapipe` 등

---

## 2) CSV 스키마(최소 계약)

실험 스크립트는 아래 **최소 컬럼**을 요구합니다.

- **필수**:  
  - `path` — 프로젝트 루트 기준 **상대경로**(예: `data_shared/cropped_faces_160/M1/img_0001.jpg`)  
  - `label_en` **또는** `orig_kor` **또는** `label` — 클래스명(문자열)
- **선택**(FaceAPI 평가 IoU 매칭에 사용):  
  - `minX,minY,maxX,maxY` — 원본 이미지 내 얼굴 박스

> 컬럼명이 다르면, **동일 의미**로 rename해서 맞춰주면 됩니다.

---

## 3) 파이프라인 개요

1. **원천→어그리게이트 CSV** (`data_raw` → `all_faces.csv`)  
   - AI Hub 메타/라벨을 읽어 **이미지 경로/라벨/박스**를 하나의 CSV로 합칩니다.
2. **크롭 생성(160×160)** (`all_faces.csv` → `data_shared/cropped_faces_160/`)  
   - 박스 기준으로 얼굴만 잘라 160×160으로 리사이즈(패딩 허용).
   - 생성된 **크롭 경로**로 `path`를 업데이트한 CSV(`all_faces_cropped.csv`)를 만듭니다.
3. **분할(M1/M2/K7)** (`all_faces_cropped.csv` → `cropped_faces_csv/*.csv`)  
   - 분할 규칙에 따라 train/val/test CSV를 생성합니다.

모든 단계는 `scripts/`의 헬퍼 스크립트 또는 아래 **대체 스니펫**으로 수행할 수 있습니다.

---

## 4) 라벨 정규화

### 4.1 K7(최종 배포)
- 원 라벨: `["기쁨","당황","분노","불안","상처","슬픔","중립"]`를 **그대로 사용**합니다.  
  (`orig_kor` 컬럼 권장)

### 4.2 M1 (6-class)
- 사용 라벨(영문): `["angry","fearful","happy","neutral","sad","surprised"]`

### 4.3 M2 (6-class)
- 사용 라벨(영문): `["angry","disgusted","fearful","happy","neutral","sad"]`

> **주의:** K7↔M1/M2는 **완전 대응이 아닙니다.**  
> M1/M2는 서비스 데이터셋 정의에 맞춘 **커스텀 6클래스 세트**입니다.  
> 따라서 **라벨 매핑 규칙**은 실험 정책에 맞춰 별도 YAML/딕셔너리로 관리하세요(예: `configs/labelmap_M1.yaml`).

---

## 5) 헬퍼 스크립트 사용(권장)

> 아래 경로/옵션은 프로젝트의 `scripts/` 폴더에 있는 헬퍼를 가정합니다.  
> 스크립트가 없다면 **6장 대체 스니펫**으로 동일 결과를 만들 수 있습니다.

### 5.1 원천 → 어그리게이트 CSV

```bash
python scripts/aihub_to_csv.py \
  --raw-root data_raw/aihub_korean_emotion \
  --out data_shared/all_faces.csv \
  --keep-cols path,label,orig_kor,minX,minY,maxX,maxY
```

- 출력: `data_shared/all_faces.csv`

### 5.2 얼굴 크롭(160×160) 생성

```bash
python scripts/crop_faces_160.py \
  --csv data_shared/all_faces.csv \
  --dst-dir data_shared/cropped_faces_160 \
  --out-csv data_shared/all_faces_cropped.csv \
  --img-size 160 --square-pad reflect --quality 95
```

- 출력 이미지: `data_shared/cropped_faces_160/...`
- 출력 CSV: `data_shared/all_faces_cropped.csv` (`path`가 크롭 파일로 업데이트)

### 5.3 분할(M1/M2/K7) CSV 생성

```bash
# K7: 7클래스 그대로, stratified split
python scripts/make_splits.py \
  --csv data_shared/all_faces_cropped.csv \
  --scheme K7 --val-ratio 0.10 --test-ratio 0.20 --seed 1337 \
  --out-dir data_shared/cropped_faces_csv

# M1: 6클래스(영문)로 매핑 후 split
python scripts/make_splits.py \
  --csv data_shared/all_faces_cropped.csv \
  --scheme M1 --labelmap configs/labelmap_M1.yaml \
  --val-ratio 0.10 --test-ratio 0.20 --seed 1337 \
  --out-dir data_shared/cropped_faces_csv

# M2: 6클래스(영문)로 매핑 후 split
python scripts/make_splits.py \
  --csv data_shared/all_faces_cropped.csv \
  --scheme M2 --labelmap configs/labelmap_M2.yaml \
  --val-ratio 0.10 --test-ratio 0.20 --seed 1337 \
  --out-dir data_shared/cropped_faces_csv
```

- 출력: `train_*.csv / val_*.csv / test_*.csv` (모두 **프로젝트 루트 기준 상대경로**)

---

## 6) (대체) 스니펫으로 빠르게 만들기

> 스크립트가 아직 없다면, 아래 한 파일씩 실행해서 **최소한의 실험 계약**을 맞출 수 있습니다.  
> (AI Hub 주석 포맷은 배포본마다 다를 수 있으니, 경로/키는 상황에 맞게 수정하세요.)

### 6.1 어그리게이트 CSV 만들기

```bash
python - <<'PY'
import os, json, csv, glob
RAW = "data_raw/aihub_korean_emotion"
OUT = "data_shared/all_faces.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

rows=[]
# 예시: annotations/*.json 에 [ { "image_path": "...", "label": "...", "bbox":[x,y,w,h] }, ... ]
for ann in glob.glob(os.path.join(RAW,"annotations","*.json")):
    with open(ann,"r",encoding="utf-8") as f:
        data=json.load(f)
    for it in data:
        p = it.get("image_path") or it.get("path")
        lab = it.get("label") or it.get("orig_kor")
        box = it.get("bbox") or [it.get("minX"),it.get("minY"),it.get("w"),it.get("h")]
        if p is None or lab is None: continue
        p = p.replace("\\","/")
        minX,minY = box[0], box[1]
        if len(box)==4: maxX, maxY = minX+box[2], minY+box[3]
        else: maxX,maxY = it.get("maxX"), it.get("maxY")
        rows.append(dict(path=p,label=lab,orig_kor=lab,minX=minX,minY=minY,maxX=maxX,maxY=maxY))

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["path","label","orig_kor","minX","minY","maxX","maxY"])
    w.writeheader(); w.writerows(rows)
print("Wrote", OUT, "rows:", len(rows))
PY
```

### 6.2 160×160 크롭 생성(+ CSV 업데이트)

```bash
python - <<'PY'
import os, csv
from PIL import Image
import numpy as np

CSV_IN  = "data_shared/all_faces.csv"
CSV_OUT = "data_shared/all_faces_cropped.csv"
DST_DIR = "data_shared/cropped_faces_160"
SIZE    = 160

os.makedirs(DST_DIR, exist_ok=True)
rows_out=[]
with open(CSV_IN,"r",encoding="utf-8") as f:
    r=csv.DictReader(f)
    for i,row in enumerate(r):
        src=row["path"].replace("\\","/")
        if not os.path.isabs(src):
            src=os.path.join(".",src)
        try:
            im=Image.open(src).convert("RGB")
        except:
            # 손상 이미지 → 검은 패치
            im=Image.fromarray(np.zeros((SIZE,SIZE,3),dtype=np.uint8))
            dst_rel=os.path.join(DST_DIR,f"bad_{i:07d}.jpg").replace("\\","/")
            im.save(dst_rel,quality=95)
            row["path"]=dst_rel
            rows_out.append(row); continue
        try:
            x1,y1,x2,y2=map(float,[row["minX"],row["minY"],row["maxX"],row["maxY"]])
            x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(im.width,int(x2)); y2=min(im.height,int(y2))
            face=im.crop((x1,y1,x2,y2))
        except:
            face=im
        # 정사각 패딩 후 리사이즈
        w,h=face.size; L=max(w,h)
        canvas=Image.new("RGB",(L,L), (0,0,0))
        canvas.paste(face, ((L-w)//2,(L-h)//2))
        face=canvas.resize((SIZE,SIZE), Image.BILINEAR)
        dst_rel=os.path.join(DST_DIR,f"crop_{i:07d}.jpg").replace("\\","/")
        os.makedirs(os.path.dirname(dst_rel), exist_ok=True)
        face.save(dst_rel,quality=95)
        row["path"]=dst_rel
        rows_out.append(row)

with open(CSV_OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows_out[0].keys())
    w.writeheader(); w.writerows(rows_out)
print("Wrote", CSV_OUT, "rows:", len(rows_out))
PY
```

### 6.3 분할(M1/M2/K7) CSV 생성(계약 충족: stratified)

```bash
python - <<'PY'
import os, csv, json, random
from collections import defaultdict
from math import floor

random.seed(1337)
CSV_IN = "data_shared/all_faces_cropped.csv"
OUTDIR = "data_shared/cropped_faces_csv"
os.makedirs(OUTDIR, exist_ok=True)

def read_rows(p):
    import csv
    with open(p,"r",encoding="utf-8") as f:
        return list(csv.DictReader(f))

rows = read_rows(CSV_IN)

def split_stratified(rows, labkey="label", val_ratio=0.10, test_ratio=0.20):
    by=defaultdict(list)
    for r in rows: by[r[labkey]].append(r)
    train,val,test=[],[],[]
    for k,v in by.items():
        n=len(v); n_test=floor(n*test_ratio); n_val=floor(n*val_ratio)
        random.shuffle(v)
        test.extend(v[:n_test]); val.extend(v[n_test:n_test+n_val]); train.extend(v[n_test+n_val:])
    return train,val,test

# K7: orig_kor 그대로
tr,val,te = split_stratified(rows, labkey="orig_kor")
for name,data in [("train_K7.csv",tr),("val_K7.csv",val),("test_K7.csv",te)]:
    with open(os.path.join(OUTDIR,name),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)

# M1: 6-class(영문) — 예시 매핑(필요시 정책에 맞게 수정)
# *주의*: 실제 서비스 정책 라벨매핑과 동일하게 맞추세요.
map_M1 = {
    "기쁨":"happy","분노":"angry","불안":"fearful","슬픔":"sad","중립":"neutral","당황":"surprised",
    "상처":None # 제외
}
m1=[r for r in rows if map_M1.get(r["orig_kor"]) is not None]
for r in m1: r["label_en"]=map_M1[r["orig_kor"]]
tr,val,te = split_stratified(m1, labkey="label_en")
for name,data in [("train_M1.csv",tr),("val_M1.csv",val),("test_M1.csv",te)]:
    with open(os.path.join(OUTDIR,name),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)

# M2: 6-class(영문) — 예시 매핑(필요시 정책에 맞게 수정)
map_M2 = {
    "기쁨":"happy","분노":"angry","불안":"fearful","슬픔":"sad","중립":"neutral","상처":"disgusted",
    "당황":None # 제외
}
m2=[r for r in rows if map_M2.get(r["orig_kor"]) is not None]
for r in m2: r["label_en"]=map_M2[r["orig_kor"]]
tr,val,te = split_stratified(m2, labkey="label_en")
for name,data in [("train_M2.csv",tr),("val_M2.csv",val),("test_M2.csv",te)]:
    with open(os.path.join(OUTDIR,name),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)

print("Done. Wrote K7/M1/M2 splits to", OUTDIR)
PY
```

> 위 매핑은 **예시**입니다. 실제 서비스/실험 정책에 맞춰 조정하세요.  
> (우리 실험 결과표의 라벨 세트와 정확히 일치하도록 최종 매핑을 고정하는 것이 중요합니다.)

---

## 7) 무결성 체크(빠른 검증)

```bash
python - <<'PY'
import os, csv, collections
def stat(p, labkey):
    with open(p,"r",encoding="utf-8") as f:
        r=list(csv.DictReader(f))
    cnt=collections.Counter([x[labkey] for x in r])
    print(os.path.basename(p), "n=",len(r), "labs=", sorted(cnt.items(), key=lambda x:-x[1])[:10])
base="data_shared/cropped_faces_csv"
for n,k in [("train_K7.csv","orig_kor"),("val_K7.csv","orig_kor"),("test_K7.csv","orig_kor"),
            ("train_M1.csv","label_en"),("val_M1.csv","label_en"),("test_M1.csv","label_en"),
            ("train_M2.csv","label_en"),("val_M2.csv","label_en"),("test_M2.csv","label_en")]:
    p=os.path.join(base,n)
    if os.path.exists(p): stat(p,k)
PY
```

---

## 8) FaceAPI 평가를 위한 주의

- `eval_faceapi_fast.mjs` 는 **원본 이미지** 경로와 **박스 컬럼**(`minX..maxX`)을 활용해  
  **IoU≥0.30** 기준으로 매칭된 샘플만 **Top-1 Accuracy**를 집계합니다.  
- **크롭만 있는 CSV**로는 매칭이 불가능하므로, FaceAPI를 돌릴 CSV에는 **박스 컬럼**이 있어야 합니다.  
  (위 6.1 단계에서 **박스 컬럼을 유지**한 CSV를 만들어 두세요.)

---

## 9) 흔한 이슈

- **경로 문제**: `path`는 반드시 **프로젝트 루트 기준 상대경로**로 통일.  
- **라벨 순서**: 학습 산출물 `classes.json`의 라벨 순서가 프런트 매핑과 **동일**해야 함.  
- **클래스 불균형**: macro-F1을 주지표로 사용, oversample/CB-loss 등 고려.  
- **크롭 품질**: 작은 박스/블러/가림 비율이 높은 샘플 필터링 옵션을 스크립트에 추가 가능.

---

## 10) 다음 읽을거리

- `docs/runbook.md` — 이 CSV/크롭을 사용해 학습/평가 실행  
- `docs/faceapi_setup.md` — Windows/Node 18.20.4 환경에서 FaceAPI 평가  
- `docs/architecture.md` — 모델 구조/입출력/라벨 규약
