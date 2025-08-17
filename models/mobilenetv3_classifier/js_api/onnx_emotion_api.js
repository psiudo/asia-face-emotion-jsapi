// [FILEPATH] models/mobilenetv3_classifier/js_api/onnx_emotion_api.js
import * as ort from 'onnxruntime-web';

const IMG  = 160;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

// KAIST 7라벨(한글 고정)
const DEFAULT_KEYS = ['기쁨','당황','분노','불안','상처','슬픔','중립'];

// 필요 시 false
const DEBUG = true;

export class EmotionAPI {
  /**
   * @param {Object} cfg
   * @param {string} cfg.modelUrl
   * @param {string=} cfg.classesUrl   // {"classes":[...]} (한글 7라벨). remap은 무시
   * @param {"wasm"|"webgl"=} cfg.provider
   * @param {string=} cfg.wasmBasePath // dist/ 경로. 끝 슬래시 자동 보정
   */
  constructor(cfg) {
    this.modelUrl   = cfg.modelUrl;
    this.classesUrl = cfg.classesUrl ?? null;
    this.provider   = cfg.provider ?? 'wasm';

    if (cfg.wasmBasePath) {
      const p = cfg.wasmBasePath.endsWith('/') ? cfg.wasmBasePath : (cfg.wasmBasePath + '/');
      // @ts-ignore
      ort.env.wasm.wasmPaths = p;
    }

    this.outputKeys = DEFAULT_KEYS.slice(); // 출력 라벨 키
    this.remap      = null;                 // KAIST: remap 사용 안 함

    // 내부 전처리 캔버스
    this.canvas = document.createElement('canvas');
    this.canvas.width = IMG; this.canvas.height = IMG;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2D context unavailable');
    this.ctx = ctx;

    this.session    = null;
    this.inputName  = null;
    this.outputName = null;
  }

  async init() {
    // 클래스 메타(있으면 한글 7라벨로 교체). remap은 무시.
    if (this.classesUrl) {
      try {
        const meta = await (await fetch(this.classesUrl)).json();
        if (Array.isArray(meta?.classes) && meta.classes.length === 7) {
          this.outputKeys = meta.classes.slice();
        }
      } catch (e) {
        if (DEBUG) console.warn('[warn] classesUrl fetch failed:', e);
      }
    }

    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: [this.provider],
      graphOptimizationLevel: 'all',
    });

    this.inputName  = this.session.inputNames?.[0]  ?? null;
    this.outputName = this.session.outputNames?.[0] ?? null;
    if (!this.inputName) throw new Error('ONNX input name unresolved');
    if (DEBUG) console.log('[dbg] io names:', this.inputName, this.outputName);
  }

  _softmax(arr) {
    let m=-Infinity; for (let i=0;i<arr.length;i++) if (arr[i]>m) m=arr[i];
    let s=0; const exps=new Float32Array(arr.length);
    for (let i=0;i<arr.length;i++){ const v=Math.exp(arr[i]-m); exps[i]=v; s+=v; }
    for (let i=0;i<arr.length;i++) exps[i]/=s;
    return exps;
  }

  _extractCHW(src, box) {
    const w = 'videoWidth'  in src ? src.videoWidth  : ('width'  in src ? src.width  : IMG);
    const h = 'videoHeight' in src ? src.videoHeight : ('height' in src ? src.height : IMG);
    let sx=0, sy=0, sw=w, sh=h;
    if (box) {
      sx = Math.max(0, box.x|0);
      sy = Math.max(0, box.y|0);
      sw = Math.max(1, box.width|0);
      sh = Math.max(1, box.height|0);
    }
    this.ctx.clearRect(0,0,IMG,IMG);
    this.ctx.drawImage(src, sx, sy, sw, sh, 0, 0, IMG, IMG);
    const data = this.ctx.getImageData(0,0,IMG,IMG).data;

    const out = new Float32Array(3*IMG*IMG);
    let p=0;
    for (let i=0;i<IMG*IMG;i++){
      const r = data[p++]/255, g = data[p++]/255, b = data[p++]/255; p++; // skip A
      out[i]             = (r - MEAN[0]) / STD[0];
      out[i + IMG*IMG]   = (g - MEAN[1]) / STD[1];
      out[i + 2*IMG*IMG] = (b - MEAN[2]) / STD[2];
    }
    return out;
  }

  _pick7(outMap) {
    // 출력 맵에서 길이 7 텐서 탐색
    let candKey = null, cand = null;
    for (const k of Object.keys(outMap)) {
      const t = outMap[k];
      const len = t?.data?.length ?? 0;
      const dims = t?.dims || [];
      if (len === 7) { candKey = k; cand = t; break; }
      if (!cand && dims.length && dims[dims.length-1] === 7) { candKey = k; cand = t; }
    }
    return cand ? { key: candKey, tensor: cand } : null;
  }

  async _runFloatCHW(chw) {
    if (!this.session) throw new Error('init() not called');

    if (DEBUG) {
      const n = IMG*IMG; let r=0,g=0,b=0;
      for (let i=0;i<n;i++){ r+=chw[i]; g+=chw[i+n]; b+=chw[i+2*n]; }
      console.log('[dbg] mean(R,G,B)=', (r/n).toFixed(4), (g/n).toFixed(4), (b/n).toFixed(4));
    }

    const input = new ort.Tensor('float32', chw, [1,3,IMG,IMG]);
    const outMap = await this.session.run({ [this.inputName]: input });

    // 선언된 outputName 우선
    let out = this.outputName ? outMap[this.outputName] : null;
    // 대안: 7채널 자동 선택
    if (!out || !out.data || (out.data.length !== 7 && !(out.dims && out.dims[out.dims.length-1] === 7))) {
      const pick = this._pick7(outMap);
      if (!pick) throw new Error('no 7-channel output found. outputs=' + Object.keys(outMap).join(','));
      out = pick.tensor;
      this.outputName = pick.key;
      if (DEBUG) console.log('[dbg] picked output:', this.outputName);
    }

    const { dims, data } = out;
    if (DEBUG) console.log('[dbg] output dims=', dims, 'len=', data.length);
    if (data.length !== 7) throw new Error(`unexpected logits length=${data.length}`);

    const probs = this._softmax(data);
    const scores = {};
    for (let i=0;i<7;i++){
      const key = this.outputKeys[i] ?? `C${i}`;
      // KAIST: remap 미사용 → i 그대로
      scores[key] = Number(probs[i].toFixed(6));
    }
    if (DEBUG) {
      const top3 = Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0,3);
      console.log('[dbg] top3=', top3);
    }
    return scores;
  }

  async predictFromVideo(video, box){   return this._runFloatCHW(this._extractCHW(video,  box)); }
  async predictFromImage(image, box){   return this._runFloatCHW(this._extractCHW(image,  box)); }
  async predictFromCanvas(canvas, box){ return this._runFloatCHW(this._extractCHW(canvas, box)); }
}
