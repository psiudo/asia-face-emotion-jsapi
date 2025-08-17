<!-- [FILEPATH] docs/runbook.md -->
# Runbook — FaceAPI vs MobileNetV3 (M1/M2/K7)

**목적:** 준비된 CSV/이미지로 곧장 **학습/평가**하고, **결과 집계/정리**까지 끝내기.  
설치는 `docs/faceapi_setup.md`, 구조 설명은 `docs/architecture.md` 참고.

---

## 0) 선행: 데이터 준비

- 원천 데이터 다운로드/정리/크롭/분할 절차는 **`docs/data_prep.md`** 를 먼저 수행하세요.  
  (최종 산출물: `data_shared/cropped_faces_csv/*.csv`, `data_shared/cropped_faces_160/`)

---

## 1) 실험 의도 & 데이터 분할

- **M1, M2**: 수집 시기/도메인/잡음이 다른 **두 독립 분할**.  
  각 분할에서 **probe(헤드만)** / **finetune(스테이지드 언프리즈)** 모두 측정.
- **K7**: 최종 배포용 **7클래스 통합 분할** → **finetune만** 수행.

총 7개 결과:
- FaceAPI: M1, M2 (매칭 Top-1 Acc)  
- MobileNetV3: M1-Probe, M1-Finetune, M2-Probe, M2-Finetune, K7-Finetune

---

## 2) 폴더 규약(프로젝트 루트 기준)

```
Asia_Face_Emtion_Improve/
├─ data_shared/
│  ├─ cropped_faces_csv/
│  └─ cropped_faces_160/
├─ models/
│  ├─ faceapi_baseline/
│  └─ mobilenetv3_classifier/
└─ docs/
```

> `train_from_csv.py`는 CSV의 `path`를 **프로젝트 루트 기준 상대경로**로 해석.  
> 실행 시 항상 `--project-root .` 옵션을 넣어 경로 보정 고정.

---

## 3) CSV 스키마

- **필수**: `path`, 그리고 `label_en` **또는** `orig_kor` **또는** `label`  
- **선택**: `minX,minY,maxX,maxY`(FaceAPI 평가지표 IoU 계산용)

---

## 4) 모델·레이어 전략(MobileNetV3-Small)

- 백본: `torchvision.models.mobilenet_v3_small`(ImageNet 사전학습)
- 헤드 교체: `classifier[-1] = Linear(in_features, num_classes)`
- **Probe**: 백본 동결, 헤드만  
- **Finetune**: 스테이지드 언프리즈  
  1) 헤드만 → 2) 마지막 2블록+헤드 → 3) 전층
- AMP: CUDA 시 자동(`torch.amp.autocast` + `GradScaler`)
- Seed: 기본 1337

---

## 5) 실행 명령(복붙)

> **전제**: 프로젝트 루트에서 실행. VRAM 부족 시 `--batch` 축소.

### 5.1 MobileNet — M1 (Probe → Finetune 30)
```bash
python models/mobilenetv3_classifier/training/train_from_csv.py --mode probe    --epochs 5  --batch 1024 --workers 8 --lr-head 7e-4 \
  --train-csv data_shared/cropped_faces_csv/train_M1.csv --val-csv data_shared/cropped_faces_csv/val_M1.csv --test-csv data_shared/cropped_faces_csv/test_M1.csv \
  --out models/mobilenetv3_classifier/runs/probe_M1_160 --project-root . --verbose

python models/mobilenetv3_classifier/training/train_from_csv.py --mode finetune --epochs 30 --batch 1024 --workers 8 --lr-backbone 1e-4 --lr-head 5e-4 \
  --train-csv data_shared/cropped_faces_csv/train_M1.csv --val-csv data_shared/cropped_faces_csv/val_M1.csv --test-csv data_shared/cropped_faces_csv/test_M1.csv \
  --out models/mobilenetv3_classifier/runs/finetune_M1_160 --project-root . --verbose
```

### 5.2 MobileNet — M2 (Probe → Finetune 30)
```bash
python models/mobilenetv3_classifier/training/train_from_csv.py --mode probe    --epochs 5  --batch 1024 --workers 8 --lr-head 7e-4 \
  --train-csv data_shared/cropped_faces_csv/train_M2.csv --val-csv data_shared/cropped_faces_csv/val_M2.csv --test-csv data_shared/cropped_faces_csv/test_M2.csv \
  --out models/mobilenetv3_classifier/runs/probe_M2_160 --project-root . --verbose

python models/mobilenetv3_classifier/training/train_from_csv.py --mode finetune --epochs 30 --batch 1024 --workers 8 --lr-backbone 1e-4 --lr-head 5e-4 \
  --train-csv data_shared/cropped_faces_csv/train_M2.csv --val-csv data_shared/cropped_faces_csv/val_M2.csv --test-csv data_shared/cropped_faces_csv/test_M2.csv \
  --out models/mobilenetv3_classifier/runs/finetune_M2_160 --project-root . --verbose
```

### 5.3 MobileNet — K7 (Finetune 50)
```bash
python models/mobilenetv3_classifier/training/train_from_csv.py --mode finetune --epochs 50 --batch 1024 --workers 8 --lr-backbone 1e-4 --lr-head 5e-4 \
  --train-csv data_shared/cropped_faces_csv/train_K7.csv --val-csv data_shared/cropped_faces_csv/val_K7.csv --test-csv data_shared/cropped_faces_csv/test_K7.csv \
  --out models/mobilenetv3_classifier/runs/finetune_K7_160 --project-root . --verbose
```

### 5.4 FaceAPI — M1/M2 평가 (매칭 Top-1, CPU)
> **중요**: FaceAPI 입력은 **splits_csv**(원본 경로 + bbox 포함)입니다.
```bash
cd models/faceapi_baseline
node scripts/eval_faceapi_fast.mjs ../../data_shared/splits_csv/test_M1.csv models results/faceapi_M1_eval.csv
node scripts/eval_faceapi_fast.mjs ../../data_shared/splits_csv/test_M2.csv models results/faceapi_M2_eval.csv
```

---

## 6) 산출물(경로/파일)

### MobileNet(out 디렉토리)
```
models/mobilenetv3_classifier/runs/<EXP_NAME>/
├─ best.pt
├─ val_metrics_best.json
├─ test_metrics.json
└─ test_preds.csv   # [path,true,pred]
```

### FaceAPI 평가 (splits_csv 사용)
```
models/faceapi_baseline/results/
├─ faceapi_M1_eval.csv
└─ faceapi_M2_eval.csv
# 컬럼: image,gt_label,pred_label,score,face_detected,iou,correct
# 지표: matched(IoU≥0.30 & face_detected=1) Top-1 Acc
```

### MobileNet 학습 (cropped_faces_csv 사용)
```
data_shared/cropped_faces_csv/
├─ train_*.csv / val_*.csv / test_*.csv  # path: 크롭 이미지 경로
```


---

## 7) 결과 요약(실측 반영)

> FaceAPI는 **매칭 샘플 한정**입니다. (두 분할 모두 매칭 커버리지 **50.08%**)

| Split | 방법 | Epochs | Batch | macro-F1 (val best) | macro-F1 (test) | Top-1 Acc (test) | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| M1 | FaceAPI (matched) | – | – | – | – | **42.07%** | IoU≥0.30 & detected=1 |
| M1 | MobileNet Probe | 5 | 1024 | **0.4617** | **0.4597** | **50.34%** | 헤드만 |
| M1 | MobileNet Finetune | 30 | 1024 | **0.7303** | **0.7298** | **74.94%** | 스테이지드 |
| M2 | FaceAPI (matched) | – | – | – | – | **32.09%** | IoU 기준 동일 |
| M2 | MobileNet Probe | 5 | 1024 | **0.4590** | **0.4552** | **49.90%** | 〃 |
| M2 | MobileNet Finetune | 30 | 1024 | **0.7300** | **0.7290** | **74.27%** | 〃 |
| K7 | MobileNet Finetune | 50 | 1024 | **0.6555** | **0.6546** | **68.65%** | 최종 후보 |

---

## 8) 빠른 집계 코드(선택)

```bash
python - <<'PY'
import json, csv, os
def load_json(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

def mobilenet_metrics(exp_dir):
    v = load_json(os.path.join(exp_dir,'val_metrics_best.json'))
    t = load_json(os.path.join(exp_dir,'test_metrics.json'))
    return round(v.get('macro_f1',0),4), round(t.get('macro_f1',0),4)

def faceapi_top1(csv_path):
    tot = 0; corr = 0; matched = 0
    with open(csv_path,'r',encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            fd = str(row.get('face_detected','0')).lower() in ('1','true')
            iou = float(row.get('iou',0))
            is_match = fd and iou >= 0.30
            tot += 1
            if is_match:
                matched += 1
                corr += int(row.get('correct','0'))
    acc = (corr/matched*100) if matched>0 else 0.0
    cov = (matched/tot*100) if tot>0 else 0.0
    return acc, cov

base = 'models/mobilenetv3_classifier/runs'
pairs = [
 ('M1 Probe',    f'{base}/probe_M1_160'),
 ('M1 Finetune', f'{base}/finetune_M1_160'),
 ('M2 Probe',    f'{base}/probe_M2_160'),
 ('M2 Finetune', f'{base}/finetune_M2_160'),
 ('K7 Finetune', f'{base}/finetune_K7_160'),
]
for name, d in pairs:
    if os.path.exists(d):
        v,t = mobilenet_metrics(d)
        print(f'{name:14s}  val_macroF1={v:.4f}  test_macroF1={t:.4f}')

fa_dir = 'models/faceapi_baseline/results'
for split in ['M1','M2']:
    p = os.path.join(fa_dir, f'faceapi_{split}_eval.csv')
    if os.path.exists(p):
        acc, cov = faceapi_top1(p)
        print(f'FaceAPI {split}: matched Top-1 Acc = {acc:.2f}% (coverage {cov:.2f}%)')
PY
```

---

## 9) 트러블슈팅

- **FileNotFoundError(이미지 경로)**: CSV `path`는 **루트 기준 상대경로**여야 함. `--project-root .` 사용.
- **CUDA OOM/속도저하**: `--batch` 1024→512/256, `--workers` 8→4/2.
- **손상 JPEG**: 160×160 블랙 패치 대체 후 학습 지속.
- **torch/torchvision 충돌**: 동일 env에서 버전 확인 및 재설치.
- **FaceAPI ERR_DLOPEN_FAILED**: `docs/faceapi_setup.md`의 Node 18.20.4 + DLL 복사 절차 재확인.
