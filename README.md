<!-- [FILEPATH] README.md -->
# Asia_Face_Emotion_Improve — FaceAPI Baseline vs MobileNetV3 (K7)

동양인 표정 인식 성능 보완을 위해 **face-api.js 기준선**과 **MobileNetV3(Pytorch)** 를 동일 분할(M1/M2/K7)로 비교한다.  
최종 산출물은 **K7(7라벨) MobileNetV3**의 **브라우저 JS API**와, **결과 시각화 노트북**이다.

- 원천 데이터: AI Hub 한국인 감정인식 복합 영상 데이터  
  링크: https://aihub.or.kr/aihubdata/data/view.do?pageIndex=3&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%91%9C%EC%A0%95&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=82

---

## 폴더 구조(요지)

```
data_raw/                                 # 원천 데이터(보관)
data_shared/
   ├─ json_kaist/
   ├─ splits_csv/                          # ★ 원본 경로+박스 CSV (FaceAPI 입력)
   ├─ cropped_faces_csv/                   # ★ 크롭 경로 CSV (MobileNet 입력)
   └─ cropped_faces_160/                   # 160×160 얼굴 크롭 이미지 (M1/M2/K7)
models/
  ├─ faceapi_baseline/
  │   ├─ scripts/eval_faceapi_fast.mjs     # CSV 기반 일괄 추론 & IoU 매칭 평가
  │   └─ models/                           # TinyFaceDetector / FaceExpression 가중치
  └─ mobilenetv3_classifier/
      ├─ training/train_from_csv.py        # 학습/평가
      └─ js_api/
         └─ onnx_emotion_api.js            # ★ 브라우저 JS API(분류 전용)
docs/
  ├─ architecture.md
  ├─ faceapi_setup.md
  ├─ runbook.md
  ├─ data_prep.md
  └─ js_api.md                             # ★ 브라우저 JS API(설치/경로/테스트)
reports/
  ├─ summary_overall_7exp.csv
  ├─ per_class_all_7exp.csv
  └─ (실험별) *_cm_counts.csv / *_cm_row_normalized.csv / *_per_class_metrics.csv
tests/
  └─ js_api_smoke_test.html                # ★ JS API 동작 확인 단일 HTML
Result_Visualization.ipynb                 # ★ 결과 시각화 노트북
```

- CSV 필수 컬럼
  - 경로: `path` (또는 `image|img|src` 중 하나)
  - 라벨: `label_en` 또는 `orig_kor` 또는 `label`
  - 상대경로는 프로젝트 루트(`.`) 기준

---

## 재현 경로

- 데이터 준비: `docs/data_prep.md`
- FaceAPI 평가: `docs/faceapi_setup.md` → Node 18.20.4, tfjs-node 설정 → `models/faceapi_baseline/scripts/eval_faceapi_fast.mjs`
- MobileNetV3 학습/평가: `models/mobilenetv3_classifier/training/train_from_csv.py`
- 시각화: `Result_Visualization.ipynb` 실행 → `reports/` 산출 확인
- 브라우저 JS API: `docs/js_api.md` 참고

---

## 결과 요약(7개 실험)

> MobileNet: **macro-F1 (val best/test)**, **Top-1 Acc(test)**  
> FaceAPI: **매칭 샘플(IoU≥0.3 & face_detected=1)** 한정 **Top-1 Acc(test)**  
> 매칭 커버리지(제공 CSV 기준): **50.08%**

| 실험 | 데이터 | 모드 | Macro-F1 (val best) | Macro-F1 (test) | Top-1 Acc (test) |
|---|---|---|---:|---:|---:|
| FaceAPI | M1 | baseline | – | – | 42.07% *(매칭 한정)* |
| FaceAPI | M2 | baseline | – | – | 32.09% *(매칭 한정)* |
| MobileNetV3 | M1 | probe | 0.4617 | 0.4597 | 50.34% |
| MobileNetV3 | M1 | finetune(30) | 0.7303 | **0.7298** | **74.94%** |
| MobileNetV3 | M2 | probe | 0.4590 | 0.4552 | 49.90% |
| MobileNetV3 | M2 | finetune(30) | 0.7300 | **0.7290** | **74.27%** |
| MobileNetV3 | K7 | finetune(50) | 0.6555 | **0.6546** | **68.65%** |

---

## 브라우저 JS API(요약)

- 목표: face-api.js의 **표정 분류 헤드**를 **K7 ONNX 분류기(JS)** 로 교체
- API 파일: `models/mobilenetv3_classifier/js_api/onnx_emotion_api.js`
- 테스트 HTML: `tests/js_api_smoke_test.html`
- 문서: `docs/js_api.md`

### 빠른 확인

```bash
npm i onnxruntime-web
python -m http.server 5173
# 브라우저: http://localhost:5173/tests/js_api_smoke_test.html
```

확인 포인트
- 업로드 이미지 → **Run Inference** → 확률 표 출력
- 콘솔:
  - `api.outputKeys` → `["기쁨","당황","분노","불안","상처","슬픔","중립"]`
  - `api.remap` → `null`/`undefined`

---

## ONNX 내보내기(필요 시)

이미 학습이 끝났고 `best.pt`가 있을 때 ONNX가 없으면 내보낸다. 스크립트가 있다면 아래를 사용.

```bash
python scripts/export_k7_onnx.py
python scripts/make_classes_k7.py
```

산출 위치(예)
```
models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx
models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json
```

> 학습 스크립트(`training/train_from_csv.py`)는 수정하지 않는다.

---

## 참고 문서

- `docs/architecture.md` — 아키텍처/입출력 규약/라벨 순서
- `docs/faceapi_setup.md` — Windows/Node 18.20.4 + tfjs-node
- `docs/runbook.md` — 실행 커맨드/집계 스니펫
- `docs/js_api.md` — JS API 설치/경로/스모크 테스트
