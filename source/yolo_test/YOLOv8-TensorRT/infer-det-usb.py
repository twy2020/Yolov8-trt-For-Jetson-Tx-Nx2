# -*- coding: utf-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import torch
import torchvision.ops as ops
from models import TRTModule
from models.utils import blob, letterbox
from config import CLASSES_DET, COLORS

def infer_stream(engine_path: str,
                 cam_index: int,
                 conf_thres: float,
                 iou_thres: float,
                 device: str):
    # 1) load TensorRT engine once
    device = torch.device(device)
    engine = TRTModule(engine_path, device)

    # 2) get model input size
    H, W = engine.inp_info[0].shape[-2:]

    # 3) open webcam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    prev_time = time.time()
    fps = 0.0

    # 4) main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # compute FPS
        cur_time = time.time()
        fps = 0.9*fps + 0.1/(cur_time - prev_time)
        prev_time = cur_time

        # letterbox + blob
        im0 = frame
        img, scale, pad = letterbox(im0, (W, H))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = blob(img_rgb, return_seg=False)
        tensor = torch.tensor(tensor, device=device)

        # inference
        with torch.no_grad():
            out = engine(tensor)  # [1, 4+nc, N]
        # decode xywh -> xyxy
        xywh = out[0, 0:4, :]
        cx, cy, w, h = xywh
        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2
        b_all = torch.stack((x1, y1, x2, y2), dim=1)

        # class scores
        nc = len(CLASSES_DET)
        cls_scores = out[0, 4:4+nc, :]
        scores_all, labels_all = cls_scores.max(0)
        labels_all = labels_all.int()

        # filter + NMS
        mask = scores_all > conf_thres
        bboxes = b_all[mask]
        scores = scores_all[mask]
        labels = labels_all[mask]
        if bboxes.numel():
            keep = ops.nms(bboxes, scores, iou_threshold=iou_thres)
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        # unpad & scale back to original
        pad_tensor = torch.tensor(pad*2, dtype=torch.float32, device=device)
        bboxes = (bboxes - pad_tensor) / scale
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy().astype(int)

        # draw
        for (box, sc, lb) in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = box.round().astype(int).tolist()
            cls = CLASSES_DET[lb]
            color = COLORS[cls]
            text = f"{cls}:{sc:.2f}"
            # box
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
            # label
            (tw, th), bl = cv2.getTextSize(text,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           0.6, 1)
            cv2.rectangle(im0,
                          (x1, y1 - th - bl),
                          (x1 + tw, y1),
                          color, -1)
            cv2.putText(im0, text, (x1, y1 - bl),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 1)

        # show FPS
        cv2.putText(im0, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # display
        cv2.imshow("Live Detection", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine",  type=str, required=True,
                   help="Path to TRT engine file")
    p.add_argument("--cam",     type=int, default=0,
                   help="Camera index, e.g. 0")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--device",  type=str, default="cuda:0",
                   help="Torch device")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_stream(args.engine, args.cam,
                 args.conf, args.iou,
                 args.device)
