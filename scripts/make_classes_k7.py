#!/usr/bin/env python3
"""
KAIST 7라벨(한국어) → face-api 7키 고정순서로 remap 생성
- 리포 루트에서 실행
- 입력:  models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json  (훈련시 사용한 클래스 "출력 인덱스" 순서)
- 출력:  models/mobilenetv3_classifier/runs/finetune_K7_160/classes_k7.json

face-api 고정 키 순서:
['angry','disgusted','fearful','happy','neutral','sad','surprised']
"""
import argparse, json, os, sys

FACE = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

# 한국어 → face-api 키 매핑(고정)
KO2EN = {
    '분노': 'angry',
    '상처': 'disgusted',
    '불안': 'fearful',
    '기쁨': 'happy',
    '중립': 'neutral',
    '슬픔': 'sad',
    '당황': 'surprised',
}

def extract_labels(obj):
    """
    classes.json 형식 다양한 경우 지원:
    - ["기쁨","당황",...]
    - {"classes": [...]} 또는 {"labels":[...]}
    - [{"name":"기쁨"}, {"name":"당황"}, ...] 등 dict 목록
    """
    if isinstance(obj, dict):
        arr = obj.get('classes') or obj.get('labels')
    else:
        arr = obj
    if not isinstance(arr, list):
        raise ValueError("classes.json에서 클래스 배열을 찾지 못함")

    out = []
    for it in arr:
        if isinstance(it, str):
            out.append(it.strip())
        elif isinstance(it, dict):
            val = (it.get('orig_kor') or it.get('ko') or it.get('kor') or
                   it.get('label_kor') or it.get('label') or it.get('name'))
            if not isinstance(val, str):
                raise ValueError(f"클래스 항목 해석 실패: {it}")
            out.append(val.strip())
        else:
            raise ValueError(f"클래스 항목 형식 미지원: {type(it)}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/mobilenetv3_classifier/runs/finetune_K7_160/classes.json")
    ap.add_argument("--out", default="models/mobilenetv3_classifier/runs/finetune_K7_160/classes_k7.json")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"[ERROR] not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    with open(args.src, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ko_seq = extract_labels(meta)  # 모델 출력 인덱스 0..6의 '한국어 라벨' 순서
    if len(ko_seq) != 7:
        print(f"[ERROR] 클래스 수가 7이 아님: {len(ko_seq)} / {ko_seq}", file=sys.stderr)
        sys.exit(1)

    # 모델 인덱스(0..6) → face-api 순서 인덱스(i)의 소스 인덱스 remap[i]
    idx_by_en = {}
    for idx, ko in enumerate(ko_seq):
        en = KO2EN.get(ko)
        if en is None:
            print(f"[ERROR] 미정의 한국어 라벨: {ko}  (KO2EN 테이블 보강 필요)", file=sys.stderr)
            sys.exit(1)
        if en in idx_by_en:
            print(f"[ERROR] 중복 라벨: {ko} -> {en}", file=sys.stderr)
            sys.exit(1)
        idx_by_en[en] = idx

    remap = []
    for en in FACE:
        if en not in idx_by_en:
            print(f"[ERROR] 매핑 누락: {en}", file=sys.stderr)
            sys.exit(1)
        remap.append(idx_by_en[en])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"classes": FACE, "remap": remap}, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {args.out}")
    print(f"[INFO] model(ko) order : {ko_seq}")
    print(f"[INFO] FACE(en) order  : {FACE}")
    print(f"[INFO] remap           : {remap}")

if __name__ == "__main__":
    main()
