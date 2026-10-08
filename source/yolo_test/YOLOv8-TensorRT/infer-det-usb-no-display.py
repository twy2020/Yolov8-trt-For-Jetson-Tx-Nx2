# -*- coding: utf-8 -*-
import linecache

_orig_getline = linecache.getline
def _safe_getline(filename, lineno, module_globals=None):
    try:
        return _orig_getline(filename, lineno, module_globals)
    except Exception:
        return ""
linecache.getline = _safe_getline

import argparse
import time

import cv2
import torch
import torchvision.ops as ops

from models import TRTModule
from models.utils import blob, letterbox
from config import CLASSES_DET

def infer_no_display(engine_path: str,
                     cam_index: int,
                     conf_thres: float,
                     iou_thres: float,
                     device: str):
    # 1) 加载 TensorRT 引擎
    device = torch.device(device)
    engine = TRTModule(engine_path, device)
    H, W = engine.inp_info[0].shape[-2:]

    # 2) 打开摄像头
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算并平滑 fps
            now = time.time()
            instantaneous_fps = 1.0 / (now - prev_time)
            fps = fps * 0.9 + instantaneous_fps * 0.1
            prev_time = now

            # 3) 预处理：letterbox → blob → tensor
            img, scale, pad = letterbox(frame, (W, H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr = blob(img, return_seg=False)        # numpy [1,3,H,W]
            inp = torch.tensor(arr, device=device)

            # 4) 推理
            with torch.no_grad():
                out = engine(inp)                    # [1, 4+nc, N]

            # 5) 解码 bbox xywh -> xyxy
            xywh = out[0, 0:4, :]                   # [4, N]
            cx, cy, w, h = xywh
            x1 = cx - w/2; y1 = cy - h/2
            x2 = cx + w/2; y2 = cy + h/2
            b_all = torch.stack((x1, y1, x2, y2), dim=1)  # [N,4]

            # 6) 取 class scores
            nc = len(CLASSES_DET)
            cls_scores = out[0, 4:4+nc, :]          # [nc, N]
            scores_all, labels_all = cls_scores.max(0)  # [N], [N]
            labels_all = labels_all.int()

            # 7) 置信度过滤
            mask = scores_all > conf_thres
            bboxes = b_all[mask]
            scores = scores_all[mask]
            labels = labels_all[mask]

            # 8) NMS
            if bboxes.numel():
                keep = ops.nms(bboxes, scores, iou_threshold=iou_thres)
                bboxes = bboxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # 9) 反 padding & scale 回原图
            pad_tensor = torch.tensor(pad * 2,
                                      dtype=torch.float32,
                                      device=device)
            bboxes = (bboxes - pad_tensor) / scale

            # 10) 转 CPU + numpy，准备打印
            bboxes = bboxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy().astype(int)

            # 11) 打印本帧检测结果和 FPS
            dets = []
            for (box, sc, lb) in zip(bboxes, scores, labels):
                x1_i, y1_i, x2_i, y2_i = box.round().astype(int).tolist()
                dets.append((CLASSES_DET[lb], float(sc), x1_i, y1_i, x2_i, y2_i))
            print(f"[FPS {fps:.1f}] Detections: {dets}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def parse_args():
    p = argparse.ArgumentParser(description="Real-time TRT detection without display")
    p.add_argument("--engine", type=str, required=True,
                   help="Path to TRT engine file (.engine)")
    p.add_argument("--cam", type=int, default=0,
                   help="Camera index (e.g. 0)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IOU threshold")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Torch device for inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_no_display(args.engine,
                     args.cam,
                     args.conf,
                     args.iou,
                     args.device)
