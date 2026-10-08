# export_trt.py

from ultralytics import YOLO

if __name__ == "__main__":
    print("🚀 正在导出 sehuan_v11 为 TensorRT...")
    model = YOLO("sehuan_v11.pt")
    model.export(format="engine", device=0, half=True, dynamic=True)
    print("✅ 导出成功: sehuan_v11.engine 已生成")
