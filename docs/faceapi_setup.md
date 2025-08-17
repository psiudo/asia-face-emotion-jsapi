<!-- [FILEPATH] docs/faceapi_setup.md -->
# FaceAPI 세팅 & 평가 실행(Windows, Node 18.20.4)

**목표:** `ERR_DLOPEN_FAILED` 같은 오류를 피하면서 **M1/M2** CSV를 빠르게 평가.  
**평가 지표:** **매칭 샘플(IoU≥0.3 & `face_detected=1`)에 한정된 Top-1 Accuracy**.

> **CSV가 아직 없다면** 먼저 `docs/data_prep.md` 를 따라 `data_shared/cropped_faces_csv/*.csv` 를 만드세요.

---

## 0) 전제 조건

- OS: **Windows 10/11 x64**
- Node.js: **v18.20.4 고정** — **20/22 금지** (ABI 차이)
- Git 설치
- 프로젝트 루트: `C:\Users\<YOU>\Desktop\Asia_Face_Emtion_Improve\`

```powershell
nvm list
nvm install 18.20.4
nvm use 18.20.4
node -v
```

---

## 1) 설치 & tfjs-node 준비

```powershell
cd $env:USERPROFILE\Desktop\Asia_Face_Emtion_Improve\models\faceapi_baseline

npm config set ignore-scripts false
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item -Force package-lock.json -ErrorAction SilentlyContinue
npm cache clean --force
npm install

npm i @tensorflow/tfjs-node@4.20.0 --foreground-scripts
Copy-Item ".\node_modules\@tensorflow\tfjs-node\deps\lib\tensorflow.dll" `
          ".\node_modules\@tensorflow\tfjs-node\lib\napi-v8\" -Force

node -e "require('@tensorflow/tfjs-node'); console.log('OK')"
```

---

## 2) 평가 실행 (M1/M2)

```powershell
# 작업 폴더: models/faceapi_baseline
node scripts/eval_faceapi_fast.mjs ../../data_shared/splits_csv/test_M1.csv models results/faceapi_M1_eval.csv
node scripts/eval_faceapi_fast.mjs ../../data_shared/splits_csv/test_M2.csv models results/faceapi_M2_eval.csv
```

- 결과 CSV: `models/faceapi_baseline/results/faceapi_*.csv`
- 주요 컬럼: `image, gt_label, pred_label, score, face_detected(0/1), iou, correct(0/1)`
- **보고 지표**: 콘솔 요약의 **`Top-1 accuracy (matched)`**  
  (= IoU≥0.3 & face_detected=1로 **매칭된 샘플에 한정**)

---

## 3) 트러블슈팅

- `ERR_DLOPEN_FAILED ... tfjs_binding.node` → **tensorflow.dll 복사** 후 재시도  
- `Cannot find package '@tensorflow/tfjs-node'` → 설치 재시도  
- Node 22에서 `TextEncoder...` → **Node 18.20.4** 사용  
- `npm ci` 실패 → `npm install` 사용  
- 손상 JPEG → 스킵(로그만 남김)
```

````markdown
<!-- [FILEPATH] docs/architecture.md -->
# 얼굴 표정 분류: 기존(face-api.js) ↔ 교체(MobileNetV3-Small) 아키텍처 비교

이 문서는 **비교 실험의 두 축**인 `face-api.js`(기존)와 `MobileNetV3-Small + Softmax Head`(교체)의 **모듈/레이어 구조와 입·출력 규약**을 한눈에 정리합니다.  
추론/학습 스크립트 사용법은 `docs/runbook.md`, 데이터 준비는 `docs/data_prep.md` 를 참고하세요.

---

## 1) 기존: face-api.js 파이프라인 개요

face-api.js는 **탐지기(Detector)** + (선택) **랜드마크** + **헤드(표정/임베딩/연령성별)** 구조의 **모듈식 파이프라인**입니다.

### 1.1 상위 파이프라인
- **Detector (택1)**  
  - `TinyFaceDetector` : Tiny-YOLOv2 계열의 경량 CNN (Depthwise Separable Conv 기반)  
  - `SsdMobilenetv1`   : SSD 헤드 + MobileNet v1 백본  
  - `Mtcnn`            : PNet → RNet → ONet 3단계 Cascade
- **(선택) Landmark**  
  - `FaceLandmark68Net` / `FaceLandmark68TinyNet` (출력: 68점 × (x,y)=136 채널)
- **Heads (Task-specific)**  
  - **FaceExpressionNet**: 7표정 확률 (neutral, happy, sad, angry, fearful, disgusted, surprised)  
  - **FaceRecognitionNet**: 128차원 임베딩  
  - **AgeGenderNet**: 나이(회귀) + 성별(분류)

> 기준선 평가는 **TinyFaceDetector + FaceExpressionNet** 조합을 사용합니다.  
> 추론 후 **박스-라벨 매칭은 IoU≥0.30** 기준으로 계산합니다.

### 1.2 Detector 내부 개괄

**TinyFaceDetector (Tiny-YOLOv2 스타일)**  
- **DWConv → PWConv** 스택으로 채널을 늘리며 다운샘플.
- 헤드: 앵커 × (bbox 파라미터 + 점수) 예측 → NMS/thresholding.

**SsdMobilenetv1**  
- MobileNet v1 백본 + 다중 스케일 SSD 헤드.

**Mtcnn**  
- PNet(초기 박스/임계) → RNet(Refine) → ONet(최종 박스+5점 랜드마크).

### 1.3 Landmark & Heads

**Landmark (68점 / Tiny)**  
- 특징추출기 → GAP → FC → 136 로짓(68×2).

**FaceExpressionNet (7-class)**  
- 경량 특징추출기 출력 → 평탄화/FC → 7채널 로짓 → softmax.

**FaceRecognitionNet**  
- Residual 스택 → FC(→128) → L2-normalized 임베딩.

---

## 2) 교체: MobileNetV3-Small 분류기(토치비전)

**ImageNet 사전학습 MobileNetV3-Small** 백본 + **얇은 Linear 헤드**로 **K-way Softmax** 분류를 수행합니다. (K는 CSV 라벨에서 자동 추론)

### 2.1 입력/출력 규약
- **입력 텐서**: `Float32, N×3×160×160` (RGB)  
  정규화: `x = (x/255 - mean) / std`,  
  `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`
- **출력 텐서**: `Float32, N×K` (로짓) → softmax 후 argmax로 클래스 결정

### 2.2 레이어 상세 (torchvision `mobilenet_v3_small`)
입력 160×160 기준 주요 단계:

| Stage | 레이어(핵심 파라미터) | Out-Res | Out-Ch | Stride | NL/SE |
|---|---|---:|---:|:---:|:---:|
| Stem | Conv 3×3 | 80×80 | 16 | 2 | h-swish |
| Bneck1 | k3, exp16, out16 | 40×40 | 16 | 2 | ReLU/SE |
| Bneck2 | k3, exp72, out24 | 20×20 | 24 | 2 | ReLU |
| Bneck3 | k3, exp88, out24 | 20×20 | 24 | 1 | ReLU |
| Bneck4 | k5, exp96, out40 | 10×10 | 40 | 2 | h-swish/SE |
| Bneck5-6 | k5, exp240, out40 | 10×10 | 40 | 1 | h-swish/SE |
| Bneck7-8 | k5, exp120/144, out48 | 10×10 | 48 | 1 | h-swish/SE |
| Bneck9-11 | k5, exp288/576, out96 | 5×5 | 96 | 2/1/1 | h-swish/SE |
| Head-Conv | 1×1 Conv | 5×5 | 576 | 1 | h-swish |
| Pool | AdaptiveAvgPool2d | 1×1 | 576 | – | – |
| FC-Proj | Linear 576→1024 + Dropout | 1×1 | 1024 | – | h-swish |
| **Classifier** | **Linear 1024→K** | – | – | – | 로짓 |

**핵심 포인트**
- SE 채용 블록 다수 → **채널 주의집중 강화**  
- h-swish로 경량 대비 표현력/안정성 확보  
- 다운샘플: 160→80→40→20→10→5

### 2.3 교체 지점(우리 코드)
- `tv.models.mobilenet_v3_small(weights=IMAGENET1K_V1)`
- `model.classifier[-1] = nn.Linear(1024, K)`
- **라벨 순서 규약**: 학습 산출물 `classes.json`의 **라벨 순서**가 프런트 매핑(이모지 등) 인덱스와 **동일**해야 합니다.  
  순서가 다를 경우 **프런트 매핑을 `classes.json` 기준으로 갱신**하세요.
- 모드  
  - **probe**: 백본 동결, 헤드만 학습  
  - **finetune**: 단계적 언프리즈(헤드 → 마지막 2블록+헤드 → 전층)

---

## 3) 입력 전처리와 크롭 규약(공통/차이)

- **크기**: 공통 **160×160 얼굴 크롭**(CSV/이미지 사전 준비)
- **정규화**:  
  - face-api.js: 내부 전처리(0..255 스케일, 평균 RGB 보정)  
  - MobileNetV3: **ImageNet mean/std** 정규화
- **라벨 공간**:  
  - face-api.js: 7 고정(neutral/happy/sad/angry/fearful/disgusted/surprised)  
  - MobileNetV3: **CSV 라벨 그대로**(M1/M2=6클래스, K7=7클래스)
