# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import torch
import torchvision.ops as ops

from models import TRTModule
from models.utils import blob, letterbox, path_to_list
from config import CLASSES_DET, COLORS


def debug_print(step, **kwargs):
    """打印 tensor 的 shape, dtype, min/max，以及普通变量。"""
    print(f"--- [DEBUG] {step} ---")
    for name, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            t = val
            print(f"{name}: shape={tuple(t.shape)}, "
                  f"dtype={t.dtype}, min={t.min().item():.3f}, max={t.max().item():.3f}")
        else:
            print(f"{name}: {val}")
    print()


def main(args: argparse.Namespace):
    device = torch.device(args.device)
    engine = TRTModule(args.engine, device)
    debug_print("Engine loaded",
                engine_file=args.engine,
                input_md=engine.inp_info)

    # 模型输入尺寸
    H, W = engine.inp_info[0].shape[-2:]
    debug_print("Model input size", H=H, W=W)

    # 收集图片路径
    image_paths = path_to_list(args.imgs)
    debug_print("Images", count=len(image_paths),
                samples=image_paths[:3])

    out_dir = Path(args.out_dir)
    if not args.show and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    conf_thres = 0.25
    iou_thres  = 0.45
    nc = len(CLASSES_DET)  # 类别数 = 3

    for img_path in image_paths:
        orig = cv2.imread(str(img_path))
        if orig is None:
            print(f"[WARN] can't read {img_path}")
            continue
        vis = orig.copy()

        # Letterbox 缩放 + padding
        resized, scale, pad = letterbox(orig, (W, H))
        debug_print("letterbox",
                    orig_shape=orig.shape,
                    resized_shape=resized.shape,
                    scale=scale, pad=pad)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 转 tensor
        arr = blob(rgb, return_seg=False)      # shape [1,3,H,W]
        inp = torch.tensor(arr, device=device)
        debug_print("input tensor", inp=inp)

        # pad Tensor，用于反 padding
        pad_tensor = torch.tensor(pad*2,       # (dw,dh)→(dw,dh,dw,dh)
                                  dtype=torch.float32,
                                  device=device)
        debug_print("pad tensor", pad=pad_tensor)

        # 推理
        with torch.no_grad():
            out = engine(inp)                 # [1, 4+nc, N] = [1,7,N]
        debug_print("raw output", outputs=out)

        # 看看头几个 anchor 的所有通道
        raw = out[0].cpu().numpy()            # (C, N)
        C, N = raw.shape
        print("\n[DEBUG] first 5 anchors (all channels):")
        for i in range(min(5, N)):
            print(" pred %2d:" % i,
                  ", ".join(f"{v:.3f}" for v in raw[:, i]))
        print(f"---- cx,cy,w,h = ch0–3, cls_scores = ch4–{4+nc-1} ----\n")

        # Decode bbox (xywh→xyxy)
        xywh = out[0, 0:4, :]                 # [4, N]
        cx, cy, w, h = xywh
        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2
        b_all = torch.stack((x1, y1, x2, y2), dim=1)  # [N,4]

        # 各类得分
        cls_scores = out[0, 4:4+nc, :]        # [nc, N]
        scores_all, labels_all = cls_scores.max(0)   # [N], [N]
        labels_all = labels_all.int()

        debug_print("parsed",
                    total=N,
                    boxes=b_all,
                    scores=scores_all,
                    labels=labels_all)

        # 置信度过滤
        mask = scores_all > conf_thres
        debug_print("after conf filter",
                    kept=mask.sum().item(),
                    dropped=(~mask).sum().item())
        bboxes = b_all[mask]
        scores = scores_all[mask]
        labels = labels_all[mask]

        # NMS
        if bboxes.numel():
            keep = ops.nms(bboxes, scores, iou_threshold=iou_thres)
            debug_print("after NMS", kept=keep.numel())
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        else:
            debug_print("NMS skipped", boxes=0)

        # 反 padding + 缩放回原图
        bboxes = (bboxes - pad_tensor) / scale
        debug_print("after unpad/scale", bboxes=bboxes)

        # 输出结果并画框
        print(f"\nDetections in {img_path.name}:")
        if bboxes.numel() == 0:
            print("  none\n")
        else:
            for box, sc, lb in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = box.round().int().tolist()
                cname = CLASSES_DET[int(lb)]
                print(f"  - {cname:>10}: score={sc:.3f}, "
                      f"box=[{x1},{y1},{x2},{y2}]")
            print()
            for box, sc, lb in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = box.round().int().tolist()
                cname = CLASSES_DET[int(lb)]
                color = COLORS[cname]
                txt = f"{cname}:{sc:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                (tw, th), bl = cv2.getTextSize(
                    txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(vis,
                              (x1, max(y1-th-bl, 0)),
                              (x1+tw, y1),
                              color, -1)
                cv2.putText(vis, txt,
                            (x1, max(y1-4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,255), 1)

        # 保存或显示
        if args.show:
            cv2.imshow("det", vis)
            cv2.waitKey(0)
        else:
            out_file = out_dir / img_path.name
            cv2.imwrite(str(out_file), vis)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRT detection with corrected class slicing"
    )
    parser.add_argument("--engine", required=True,
                        help="Path to TRT engine file (.engine)")
    parser.add_argument("--imgs",   required=True,
                        help="Image file or directory")
    parser.add_argument("--show", action="store_true",
                        help="Display results instead of saving")
    parser.add_argument("--out-dir", default="./output",
                        help="Directory to save output images")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device for inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
