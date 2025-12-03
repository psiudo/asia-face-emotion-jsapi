# Asia Face Emotion Improve — FaceAPI Baseline vs MobileNetV3 K7

---

## English Version

<details>
<summary>Click to expand</summary>

## Overview

This project provides an emotion classifier trained on Korean facial expression data.
It compares the default face api js expression head and a MobileNetV3 classifier on the same splits M1 M2 K7.

The final artifacts are the K7 MobileNetV3 ONNX model a browser ready JS API and a result visualization notebook.

This work starts from the bias report in face api js issue number 469:
https://github.com/justadudewhohacks/face-api.js/issues/469

The default expression head shows lower accuracy for Asian faces.
This repository provides an alternative classifier that follows the face api js input and output format so existing code can stay the same.

- Source data: AI Hub Korean Emotion Video Dataset
  https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=82

---

## Folder Structure summary

```text
data_raw/
data_shared/
   ├─ json_kaist/
   ├─ splits_csv/                          # Original path and box CSV FaceAPI input
   ├─ cropped_faces_csv/                   # Cropped path CSV MobileNet input
   └─ cropped_faces_160/                   # 160×160 face crops M1 M2 K7
models/
  ├─ faceapi_baseline/
  │   ├─ scripts/eval_faceapi_fast.mjs     # CSV batch inference and IoU based evaluation
  │   └─ models/                           # TinyFaceDetector and FaceExpression weights
  └─ mobilenetv3_classifier/
      ├─ training/train_from_csv.py        # Training and evaluation
      └─ js_api/
         └─ onnx_emotion_api.js            # Browser JS API classifier only
docs/
  ├─ architecture.md
  ├─ faceapi_setup.md
  ├─ runbook.md
  ├─ data_prep.md
  └─ js_api.md                             # JS API install paths and tests
reports/
  ├─ summary_overall_7exp.csv
  ├─ per_class_all_7exp.csv
  └─ *_cm_counts.csv and *_cm_row_normalized.csv and *_per_class_metrics.csv
tests/
  └─ js_api_smoke_test.html                # Single page JS API smoke test
Result_Visualization.ipynb                 # Result visualization notebook
```

Required CSV columns

- Path: `path` or `image` or `img` or `src`
- Label: `label_en` or `orig_kor` or `label`

All paths are relative to the project root `.`.

---

## Reproduction

Data preparation steps are in `docs/data_prep.md`.

FaceAPI evaluation uses Node 18.20.4 plus tfjs node and the script `models/faceapi_baseline/scripts/eval_faceapi_fast.mjs`.

MobileNetV3 training and evaluation use `models/mobilenetv3_classifier/training/train_from_csv.py`.

Visualization uses `Result_Visualization.ipynb` and reads metrics from the reports folder.

The browser JS API setup and usage follow `docs/js_api.md`.

---

## Results for seven experiments

| Exp | Split | Mode | Macro F1 val best | Macro F1 test | Top 1 Acc test |
|---|---|---|---:|---:|---:|
| FaceAPI | M1 | baseline | – | – | 42.07% matched only |
| FaceAPI | M2 | baseline | – | – | 32.09% matched only |
| MobileNetV3 | M1 | probe | 0.4617 | 0.4597 | 50.34% |
| MobileNetV3 | M1 | finetune 30 | 0.7303 | **0.7298** | **74.94%** |
| MobileNetV3 | M2 | probe | 0.4590 | 0.4552 | 49.90% |
| MobileNetV3 | M2 | finetune 30 | 0.7300 | **0.7290** | **74.27%** |
| MobileNetV3 | K7 | finetune 50 | 0.6555 | **0.6546** | **68.65%** |

FaceAPI matching coverage with `IoU ≥ 0.3` and `face_detected = 1` is 50.08%.

---

## Browser JS API summary

- Goal: replace the face api js expression head with a K7 ONNX classifier in JavaScript
- API file: `models/mobilenetv3_classifier/js_api/onnx_emotion_api.js`
- Test page: `tests/js_api_smoke_test.html`

Quick check

```bash
npm i onnxruntime-web
python -m http.server 5173
# open in browser
# http://localhost:5173/tests/js_api_smoke_test.html
```

In the developer console

- `api.outputKeys` should print `["기쁨","당황","분노","불안","상처","슬픔","중립"]`
- `api.remap` should be `null` or `undefined`

---

## ONNX export optional

If training is finished and `best.pt` exists and there is no ONNX file run:

```bash
python scripts/export_k7_onnx.py
python scripts/make_classes_k7.py
```

Example output paths

```text
models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx
models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json
```

---

## References

- `docs/architecture.md`
- `docs/faceapi_setup.md`
- `docs/runbook.md`
- `docs/js_api.md`

</details>

---

## 한국어 버전


## 개요

GitHub issue 469에서 보고된 것처럼 face api js는 동양인 표정 인식 성능이 낮습니다.
이 리포지토리는 face api js 기준선 모델과 MobileNetV3 분류기를 동일 조건(M1, M2, K7)에서 비교하고, 브라우저 JS API 형태의 대체 분류기를 제공합니다.

원천 데이터는 AI Hub 한국인 감정인식 복합 영상 데이터입니다.
최종 산출물은 K7 7라벨 MobileNetV3 모델과 JS API, 그리고 결과 시각화 노트북입니다.

---

## 폴더 구조 요약

```text
data_raw/                                 # 원천 데이터 보관
data_shared/
   ├─ json_kaist/
   ├─ splits_csv/                         # ★ 원본 경로+박스 CSV FaceAPI 입력
   ├─ cropped_faces_csv/                  # ★ 크롭 경로 CSV MobileNet 입력
   └─ cropped_faces_160/                  # 160×160 얼굴 크롭 이미지 M1 M2 K7
models/
  ├─ faceapi_baseline/
  │   ├─ scripts/eval_faceapi_fast.mjs    # CSV 기반 일괄 추론 및 IoU 매칭 평가
  │   └─ models/                          # TinyFaceDetector 및 FaceExpression 가중치
  └─ mobilenetv3_classifier/
      ├─ training/train_from_csv.py       # 학습 및 평가
      └─ js_api/
         └─ onnx_emotion_api.js           # ★ 브라우저 JS API 분류 전용
docs/
  ├─ architecture.md
  ├─ faceapi_setup.md
  ├─ runbook.md
  ├─ data_prep.md
  └─ js_api.md                            # ★ 브라우저 JS API 설치 경로 테스트
reports/
  ├─ summary_overall_7exp.csv
  ├─ per_class_all_7exp.csv
  └─ (실험별) *_cm_counts.csv / *_cm_row_normalized.csv / *_per_class_metrics.csv
tests/
  └─ js_api_smoke_test.html               # ★ JS API 동작 확인 단일 HTML
Result_Visualization.ipynb                # ★ 결과 시각화 노트북
```

CSV 필수 컬럼은 다음과 같습니다.

- 경로: `path` 혹은 `image` 혹은 `img` 혹은 `src`
- 라벨: `label_en` 혹은 `orig_kor` 혹은 `label`

상대 경로 기준은 프로젝트 루트 `.` 입니다.

---

## 재현 경로

데이터 준비 절차는 `docs/data_prep.md`에 정리되어 있습니다.
FaceAPI 평가는 Node 18.20.4와 tfjs node 환경에서 `models/faceapi_baseline/scripts/eval_faceapi_fast.mjs`를 실행합니다.

MobileNetV3 학습 및 평가는 `models/mobilenetv3_classifier/training/train_from_csv.py`를 사용합니다.
시각화는 `Result_Visualization.ipynb`에서 수행하며 reports 폴더의 산출물을 사용합니다.

브라우저 JS API 사용법은 `docs/js_api.md`에 정리되어 있습니다.

---

## 결과 요약 (일곱 개 실험)

> MobileNet 지표는 macro F1 val best와 macro F1 test 그리고 Top 1 Acc test이다.
> FaceAPI 지표는 `IoU ≥ 0.3` 및 `face_detected = 1` 조건을 만족하는 매칭 샘플 기준 Top 1 Acc test이다.
> 제공 CSV 기준 FaceAPI 매칭 커버리지는 50.08%이다.

| 실험 | 데이터 | 모드 | Macro F1 val best | Macro F1 test | Top 1 Acc test |
|---|---|---|---:|---:|---:|
| FaceAPI | M1 | baseline | – | – | 42.07% 매칭 한정 |
| FaceAPI | M2 | baseline | – | – | 32.09% 매칭 한정 |
| MobileNetV3 | M1 | probe | 0.4617 | 0.4597 | 50.34% |
| MobileNetV3 | M1 | finetune 30 | 0.7303 | **0.7298** | **74.94%** |
| MobileNetV3 | M2 | probe | 0.4590 | 0.4552 | 49.90% |
| MobileNetV3 | M2 | finetune 30 | 0.7300 | **0.7290** | **74.27%** |
| MobileNetV3 | K7 | finetune 50 | 0.6555 | **0.6546** | **68.65%** |

---

## 브라우저 JS API 요약

- 목표: face api js 표정 분류 헤드를 K7 ONNX 분류기 JS로 교체
- API 파일: `models/mobilenetv3_classifier/js_api/onnx_emotion_api.js`
- 테스트 HTML: `tests/js_api_smoke_test.html`
- 상세 문서: `docs/js_api.md`

### 빠른 확인

```bash
npm i onnxruntime-web
python -m http.server 5173
# 브라우저에서 다음 주소 접속
# http://localhost:5173/tests/js_api_smoke_test.html
```

확인 사항

- 업로드 이미지에서 Run Inference 버튼 실행 후 확률 표가 출력되어야 한다.
- 브라우저 콘솔에서:
  - `api.outputKeys` 값은 `["기쁨","당황","분노","불안","상처","슬픔","중립"]` 이어야 함
  - `api.remap` 값은 `null` 또는 `undefined` 이어야 함

---

## ONNX 내보내기 (선택 사항)

이미 학습이 끝난 상태에서 `best.pt` 파일만 있고 ONNX 파일이 없으면 아래 스크립트를 실행합니다.

```bash
python scripts/export_k7_onnx.py
python scripts/make_classes_k7.py
```

예상 산출 위치는 다음과 같습니다.

```text
models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx
models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json
```

`training/train_from_csv.py` 파일은 수정하지 않습니다.

---

## 참고 문서

- `docs/architecture.md`: 아키텍처와 입출력 규약, 라벨 순서
- `docs/faceapi_setup.md`: Windows 환경과 Node 18.20.4, tfjs node 설정
- `docs/runbook.md`: 실행 커맨드와 집계 스니펫
- `docs/js_api.md`: JS API 설치 경로 및 스모크 테스트

</details>
