# run_trt_camera.py

import time
import cv2
from ultralytics import YOLO

def run_trt_inference():
    print("🚀 加载 TensorRT 引擎模型...")
    model = YOLO("yolo11n.engine")

    cap = cv2.VideoCapture(0)  # 可改为视频路径

    if not cap.isOpened():
        print("❌ 摄像头无法打开")
        return

    print("📸 摄像头已打开，开始推理...")
    fps_list = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        start = time.time()
        results = model.predict(source=frame, show=False, device=0)
        end = time.time()

        annotated = results[0].plot()
        fps = 1 / (end - start)
        fps_list.append(fps)

        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv11n TensorRT", annotated)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"平均 FPS: {sum(fps_list)/len(fps_list):.2f}")

if __name__ == "__main__":
    run_trt_inference()
