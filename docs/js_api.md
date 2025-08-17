<!-- [FILEPATH] docs/js_api.md -->
# JS API — KAIST K7 MobileNetV3 (브라우저)

## 목적
- KAIST/AI Hub **K7(7라벨)** MobileNetV3-Small을 **브라우저용 JS API**로 제공한다.
- 탐지(바운딩박스)는 외부 모듈(TinyFaceDetector 등)을 사용하고, 본 API는 **분류(7클래스 softmax)** 전용이다.

---

## 파일 경로(이 리포 기준)
- API 구현: `models/mobilenetv3_classifier/js_api/onnx_emotion_api.js`
- ONNX(예상): `models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx`
- 라벨 메타: `models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json`  
  *(한글 7라벨. remap 사용 안 함)*

> 모델 파일/라벨 메타의 실제 파일명은 프로젝트 상황에 따라 다를 수 있다. 상수 경로만 맞추면 된다.

---

## 출력 라벨(고정)
`['기쁨','당황','분노','불안','상처','슬픔','중립']`

---

## API 표면

```ts
// 브라우저 ESM 환경
import { EmotionAPI } from '/models/mobilenetv3_classifier/js_api/onnx_emotion_api.js?v=kor5';

const api = new EmotionAPI({
  modelUrl:   '/models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx',
  classesUrl: '/models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json', // 한글 7라벨
  provider:   'wasm',                         // 또는 'webgl'
  wasmBasePath: '/node_modules/onnxruntime-web/dist/' // 끝 슬래시 포함
});
await api.init();

// 탐지 모델로 얻은 ROI 박스를 전달(원 해상도 픽셀 기준). 박스를 생략하면 전체 프레임을 사용.
const scores = await api.predictFromImage(imgOrCanvasOrVideo, { x, y, width, height });
// 반환 예: { 기쁨:0.81, 중립:0.10, … }  // 소수 6자리 반올림
```

### 입력 규격
- 크기: **160×160** (API 내부에서 리사이즈)
- 정규화: **ImageNet**(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- `box` 좌표 단위: **원본 영상/이미지 픽셀**

---

## 의존성 및 경로 공개

```bash
npm i onnxruntime-web
```

- ESM 모듈 경로: `/node_modules/onnxruntime-web/dist/ort.min.mjs`
- WASM 로더 경로: `/node_modules/onnxruntime-web/dist/` *(끝 슬래시 포함)*
- 정적 서버(리포 루트에서 실행): `python -m http.server 5173`

> `tests/js_api_smoke_test.html`는 import map으로 위 경로를 그대로 사용한다.

---

## 스모크 테스트(로컬)

1) 의존성 설치
```bash
npm i onnxruntime-web
```

2) 정적 서버 실행(리포 루트)
```bash
python -m http.server 5173
```

3) 브라우저에서 열기  
`http://localhost:5173/tests/js_api_smoke_test.html`

4) 확인 항목
- 이미지 업로드 → **Run Inference** → 표에 확률 출력
- DevTools 콘솔:
  - `api.outputKeys` → `["기쁨","당황","분노","불안","상처","슬픔","중립"]`
  - `api.remap` → `null` 또는 `undefined`
  - 디버그 로그(`[dbg] output dims= … len= 7`)가 보이면 정상 경로

> 캐시로 인해 옛 코드가 로드되는 경우가 있으므로, 테스트 HTML의 모듈 임포트에 `?v=…` 쿼리 파라미터를 사용한다.

---

## 트러블슈팅

- **모듈 파싱 에러(SyntaxError: HTML comments…):**  
  JS 파일 첫 줄에 `<!-- … -->` 같은 HTML 주석이 있으면 안 된다. 제거 후 다시 로드.

- **옛 코드/영문 라벨이 계속 뜸:**  
  브라우저 캐시 문제다. `onnx_emotion_api.js?v=korX`처럼 쿼리를 바꿔 강제 갱신하고, `Ctrl+F5`.

- **WASM 404:**  
  `wasmBasePath`가 실제 경로(`/node_modules/onnxruntime-web/dist/`)와 일치해야 한다. 끝 슬래시 포함.

- **angry 등 단일 라벨로 몰림:**  
  중앙 정사각형 크롭이 얼굴을 벗어난 경우다. 탐지 박스 좌표를 `predictFromImage(..., box)`로 전달해라.

- **출력 길이 7이 아님:**  
  ONNX 내보내기 시 출력 이름/차원이 달라졌을 수 있다. 현재 구현은 **7채널 텐서를 자동 탐색**해 사용한다.

---

## ONNX가 없는 경우(선택)
이미 학습이 끝났고 `best.pt`가 있을 때, 내보내기가 필요하면 아래를 사용한다(스크립트가 있을 경우).

```bash
python scripts/export_k7_onnx.py
# 산출: models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx

python scripts/make_classes_k7.py
# 산출: models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json
```

> 학습 스크립트(`training/train_from_csv.py`)는 변경하지 않는다.
