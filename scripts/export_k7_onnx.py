#!/usr/bin/env python3
"""
MobileNetV3-Small(K7) ONNX 내보내기 스크립트.
- 사용 위치: 리포 루트에서 실행
- 기본 입력: models/mobilenetv3_classifier/runs/finetune_K7_160/best.pt
- 기본 출력: models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx
"""
import argparse, sys, os, json
import torch
import torchvision as tv

def load_state_dict(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("state_dict", ck)
    # prefix 정리
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("module."):  # DDP 등
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned

def build_model(num_classes: int = 7):
    m = tv.models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/mobilenetv3_classifier/runs/finetune_K7_160/best.pt")
    p.add_argument("--out",  default="models/mobilenetv3_classifier/runs/finetune_K7_160/k7_mnv3s_160.onnx")
    p.add_argument("--img",  type=int, default=160, help="입력 해상도(H=W)")
    p.add_argument("--opset",type=int, default=13)
    p.add_argument("--classes", type=int, default=7)
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = build_model(args.classes)
    sd = load_state_dict(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={list(missing)}, unexpected={list(unexpected)}")

    model.eval()
    dummy = torch.randn(1, 3, args.img, args.img)

    input_names  = ["input"]
    output_names = ["logits"]
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model, dummy, args.out,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=args.opset
    )
    print(f"[OK] wrote ONNX: {args.out}")

if __name__ == "__main__":
    main()
