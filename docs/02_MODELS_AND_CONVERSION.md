# 模型清单与转换复现

## 1. 可恢复的模型链路

当前 ROS 节点使用的输入均为 `images: float32[1,3,320,320]`，RGB、归一化到 0–1、NCHW、114 灰色 letterbox 填充。

| 模型 | 原部署 engine（`source/ros_ws/src/bot_vision/engine_models/`） | 输出 | 类别，按索引顺序 |
|---|---|---|---|
| 色环 | `ring_320_v8_nms_simp_static.engine` | `output0: float32[1,7,2100]` | red_ring, green_ring, blue_ring |
| 物料 | `wuliao_320_v8_nms_simp_static.engine` | `output0: float32[1,8,2100]` | red_wuliao, green_wuliao, blue_wuliao, purple |

对应且已用于本次成功重建的 ONNX 位于：

```text
source/yolo_test/YOLOv8-TensorRT/yolo/ring_320_v8_nms_simp.onnx
source/yolo_test/YOLOv8-TensorRT/yolo/wuliao_320_v8_nms_simp.onnx
```

二者在 `onnx/` 子目录还有内容完全相同的副本。SHA256：

```text
ring:   b4dcd35a1b0de09bbbdba935dd722d11c0fdd0b6eeb56e2ee45463c41af45107
wuliao: 3892742c2e552140c09537698a08920c2e7e75bc63781b8268d0e14431078474
```

证据见 `verification/onnx-contracts.json`、`verification/engine-bindings.json` 和 `environment/remote_manifest.json`。原 engine 的历史构建命令/精度设置未找到；本次提供的是经实测可用的 FP16 重建方案，不声称重新生成的二进制与原 engine 逐字节一致。

## 2. 已实测的 ONNX → TensorRT engine

在 TX2 NX、L4T 32.7.6、TensorRT 8.2.1.9/CUDA 10.2 上执行。`trtexec` 默认来自 `/usr/src/tensorrt/bin/trtexec`。转换会占用 CPU/GPU 和内存；本次每套约 8–9 分钟，设备还在运行其他程序，该耗时不是性能基准。

```bash
export YOLO_TRT_ROOT="$(pwd)"
export ENGINE_OUT="$HOME/yolo_rebuilt_engines"
mkdir -p "$ENGINE_OUT"
test ! -e "$ENGINE_OUT/ring-rebuilt.engine" || { echo '输出已存在，请换新目录'; exit 1; }
test ! -e "$ENGINE_OUT/wuliao-rebuilt.engine" || { echo '输出已存在，请换新目录'; exit 1; }
set -o pipefail

/usr/src/tensorrt/bin/trtexec \
  --onnx="$YOLO_TRT_ROOT/source/yolo_test/YOLOv8-TensorRT/yolo/ring_320_v8_nms_simp.onnx" \
  --saveEngine="$ENGINE_OUT/ring-rebuilt.engine" \
  --fp16 --workspace=256 --duration=1 --iterations=1 --warmUp=0 \
  2>&1 | tee "$ENGINE_OUT/ring-build.log"

/usr/src/tensorrt/bin/trtexec \
  --onnx="$YOLO_TRT_ROOT/source/yolo_test/YOLOv8-TensorRT/yolo/wuliao_320_v8_nms_simp.onnx" \
  --saveEngine="$ENGINE_OUT/wuliao-rebuilt.engine" \
  --fp16 --workspace=256 --duration=1 --iterations=1 --warmUp=0 \
  2>&1 | tee "$ENGINE_OUT/wuliao-build.log"
```

期望日志末尾 `PASSED TensorRT.trtexec`，命令退出码 0，生成非空 engine。`--fp16` 允许内部层使用 FP16，输入/输出仍是 FP32，符合原 C++ 代码要求。`--workspace=256` 是构建工作区限制，不是整个进程总内存限制。这里使用 TensorRT 8.2 的参数，不要直接换成 TensorRT 10 的命令/API。

这是静态 320 输入，不需要添加动态 `minShapes/maxShapes`。原 C++ 分配内存时直接使用 engine 的维度，动态负维度模型不适用于该实现。构建完成后按文档 01 的方式启动，将 `engine_path` 指向新文件即可。

此次重建日志已保存于 `verification/rebuild-ring.log`、`verification/rebuild-wuliao.log`，标准错误另存同名 `.stderr`。INT64 权重转 INT32 的提示是本次出现的 TensorRT 警告，构建最终成功。

## 3. 文件名误导与不兼容的历史路线

- `yolo/*_nms_simp.onnx` 虽然文件名带 `nms`，但 ONNX 元数据实际是 `nms=False`，输出为 `[1,4+类别数,2100]`。这是当前 C++ 节点所需格式。
- 顶层 `YOLOv8-TensorRT/ring_320_v8.onnx` 的元数据为 `nms=True`、输出 `[1,300,6]`，**不能直接替换当前模型**。
- `onnx/*_no_nms.onnx` 是更早的导出版本，哈希与上述已验证 ONNX 不同；留作历史资料，不自动混用。
- 工程中的 `export-det.py` 修改检测头并导出 `num_dets/bboxes/scores/labels` 四输出；其 `build.py`、Python 推理代码有对应上游格式假设。该路线与当前 ROS 头文件的单输出解析不同，不能看到工程名相同就直接使用。
- `source/yolo_test/export_trt.py` 为历史实验脚本；曾保存的 `sehuan_v11.pt`/`.onnx`、`yolov8n.pt` 不是部署色环/物料模型的已证实源权重，未作为部署模型发布。尤其不能把 YOLO11 的 PT 当作 YOLOv8 色环权重。
- 工程原始 README 描述上游项目，优先以本备份实测接口为准。

## 4. PT → ONNX：可推导配置与缺失项

部署用 ONNX 元数据明确记载：PyTorch **2.4.1**、Ultralytics **8.3.155**、opset **12**、batch **1**、imgsz **320**、`dynamic=False`、`half=False`、`simplify=True`、`nms=False`。这说明部署 ONNX 来自较新的 PC 导出环境，而不是 Jetson `yolo36` 里的 8.0.208。

上述版本来自模型元数据及导出环境核对，不是原训练过程的完整锁定文件。训练权重与数据未提供，复现部署请优先使用仓库中匹配的 ONNX。

如果以后找回准确的两份 PT，可以在独立 PC Python 3.10 环境尝试以下导出。**此次未执行这一段，因为缺少对应 PT，不能承诺导出内容与存档 ONNX 完全一致。**

```python
from ultralytics import YOLO

for weights in ("ring_320_v8.pt", "wuliao_320_v8.pt"):
    YOLO(weights).export(
        format="onnx", imgsz=320, batch=1, opset=12,
        dynamic=False, half=False, simplify=True, nms=False,
        device="cpu",
    )
```

环境参考：torch 2.4.1、torchvision 0.19.1、ultralytics 8.3.155、onnx 1.16.1、onnxsim 0.4.36、onnxslim 0.1.34。这些版本来自现存 PC 环境及 ONNX 元数据；历史 simplify 的具体内部实现并未被完整记录。导出后必须检查输出、类别顺序，并与存档模型用相同图像做数值/检测结果比较，然后才替换部署。

未找到：部署对应 PT、训练数据、`datasets/mydata.yaml` 的原始内容、训练超参数、训练日志及评估基准。搜索范围记录在 `environment/model_search.txt` 和 `environment/pc-trained-weight-search.txt`；未搜到不等于这些文件在其他硬盘/云盘不存在。

## 5. 已保留的历史 Python 安装资料

历史安装资料在开发者的私有离线归档中保存，未随公开仓库发布，包含：

- Jetson ARM64、CPython 3.6 的 torch 1.10.0 wheel；
- ONNX 1.10.1 wheel、ONNX Runtime GPU 1.10.0 wheel；
- torchvision 0.11.1 已构建 egg；
- yolo36 的 `conda-meta` 元数据与安装后的 Ultralytics 源码；
- 实际使用的 CMake 3.22.6 安装目录。

主归档另有 Ultralytics 8.0.208 源码 tar.gz、onnx-simplifier 源码和第三方源码。历史环境用于还原旧实验，不是当前 C++ 部署的强制依赖。torchvision egg 需兼容的旧 setuptools/easy_install 或重建 wheel；不要把 egg 直接当作现代 pip wheel。此补充包不是完整离线 Python 环境，缺少的传递依赖仍需按 freeze/conda 记录获取。
