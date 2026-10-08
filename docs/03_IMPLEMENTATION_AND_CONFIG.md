# 实现、接口与配置核对

以下路径以仓库根目录为基准。当前文档对应现有视觉源码实现。

## 1. 代码入口

| 文件 | 作用 |
|---|---|
| `source/ros_ws/src/bot_vision/src/yolo_trt_node.cpp` | ROS 参数、模型热重载、订阅图像、发布检测与可视化 |
| `source/ros_ws/src/bot_vision/include/bot_vision/yolov8.hpp` | TensorRT engine 加载、显存、letterbox、GPU 推理、后处理 |
| `source/ros_ws/src/bot_vision/include/bot_vision/common.hpp` | 检测结构、TensorRT logger、CUDA 错误检查、尺寸辅助函数 |
| `source/ros_ws/src/bot_vision/src/usb_cam_node.cpp` | V4L2 相机读取，固定 `/dev/car_camera`，动态分辨率/曝光设置 |
| `source/ros_ws/src/bot_vision/cfg/YoloTRT.cfg` | 动态参数类型、默认值和范围 |
| `source/ros_ws/src/bot_vision/msg/Detection.msg` / `DetectionArray.msg` | 检测结果消息 |
| `source/ros_ws/src/bot_vision/CMakeLists.txt` | C++14、catkin、OpenCV、CUDA、TensorRT 链接 |
| `source/ros_ws/src/bot_vision/launch/` | 原整机/单检测/相机 launch，包含历史参数不一致 |
| `source/yolo_test/YOLOv8-TensorRT/` | 上游转换与推理实验代码，Git 修改快照在 `environment/git_conversion.txt` |

## 2. 实际数据链路

```text
/camera/image_raw (BGR8)
  -> resize + 114 padding -> RGB / 255 -> NCHW [1,3,320,320]
  -> TensorRT enqueueV2
  -> output0 [1, 4+nc, 2100]
  -> 选每个候选框最高分类概率 -> 按 conf_thresh 过滤
  -> 消除 padding / 映射回原图 / 坐标裁剪
  -> 每类仅保留置信度最高的一个框
  -> /detections + /detection_result
```

这里没有执行常规 IoU NMS。即使图像中存在多个同类物体，现实现最多输出每类一个。图像上的 FPS 只计 `infer()` 段，不含完整 ROS/预处理/显示耗时，不能作为端到端帧率。

## 3. 真正有效的参数和接口

节点使用私有参数（前缀 `/yolo_trt_node/`）：

| 参数 | 用途 |
|---|---|
| `engine_path` | engine 的绝对路径 |
| `class_names_str` | 以英文逗号分隔的类别名称，顺序必须与 ONNX `names` 对应 |
| `conf_thresh` | 阈值，动态参数默认 0.3，范围 0–1 |

订阅：`camera/image_raw`（根命名空间时为 `/camera/image_raw`）。发布：`detections`、`detection_result`。新增复现 launch 支持对输入话题做 remap。

服务：`/yolo_trt_node/reload_model`，类型 `std_srvs/Trigger`。此外支持 dynamic_reconfigure；当前实现每次动态配置回调都会重新初始化检测器，即使只改阈值，也会重载模型。

`DetectionArray` 包含原图 Header 和 `Detection[]`；每项有字符串 `label`、概率 `prob`、原图像素坐标 `x/y/w/h`。`x/y` 是左上角，不是中心点。

## 4. 实测发现的配置问题

### 4.1 物料模型类别配置错位

运行快照 `environment/rosparams.yaml` 显示：

```yaml
engine_path: .../wuliao_320_v8_nms_simp_static.engine
class_names: [red_wuliao, green_wuliao, blue_wuliao]
class_names_str: red_ring,green_ring,blue_ring
conf_thresh: 0.3
detection_threshold: 0.9
```

当前源码只读取 `class_names_str`，因此前三类会被标记为色环，第四类 `purple` 会因越界被跳过。`class_names` 不生效。原 launch 中的 `detection_threshold=0.9`、`nms_threshold=0.4`、`visualization_topic` 也没有被当前源码读取。

新增 `repro/vision_only.launch` 使用与源码一致的参数，并为物料模型配置四类。原 Jetson 服务、原 launch 与原源码没有被修改。

### 4.2 默认 engine 和加载行为

`YoloTRT.cfg` 的默认 engine 是不存在的历史路径 `.../ring.engine`，应始终显式提供 `engine_path`。初始化先调用 `loadParams()` 和 `initDetector()`，然后注册动态参数回调，因此缺少 `class_names_str` 时可能先打印 FATAL，随后又使用动态默认类别运行。看到节点在线不代表参数正确。

TensorRT 封装大量依赖 `assert`，不应把无效、动态维度或不同输出结构的 engine 交给它。模型无法正确加载时，节点可能发布空检测结果；验收必须检查“Model loaded successfully”和实际图像/结果链路，不能仅检查话题存在。

### 4.3 相机路径和帧率

`usb_cam_node.cpp` 固定设备路径 `/dev/car_camera`。原 `usb_cam.launch` 的 `cam_index` 不被读取。`config/udev/99-car-camera.rules` 按设备名 `UNIQUESKY_CAR_CAMERA` 创建该链接。

节点的循环频率固定 `ros::Rate(30.0)`；动态 `publish_rate` 会设置摄像头 FPS，但不会改变这个循环 Rate。图片叠加面板假定输入宽度至少 300、足够容纳面板高度，复现使用 640×480。

### 4.4 CMake install 与 package.xml

原 CMake 的消息安装写成 `install(FILES Detection.msg DetectionArray.msg ...)`，但文件实际在 `msg/` 下；普通 catkin 编译通过不表示 `catkin_make install` 可用。本文使用 devel 空间。

`package.xml` 有重复依赖、`OpenCV/opencv2` 命名及 TODO 许可信息。构建通过依赖的是机器上现有 SDK 与显式 CMake 查找。迁移时先按环境文档安装实际依赖，不要只依赖自动 rosdep 推断。

## 5. 整机相关依赖

原 `yolo.launch` 还启动：

- `bot_tools/ros_stm.py`：STM32 串口 `/dev/ttyUpperBoard`，115200；
- `bot_tools/code128.py`：条码任务解析；
- `bot_arm/arm_communication_node.py`、`arm_control_node.py`、`detection_task_node.py`；
- `usb_cam_node`、`rqt_image_view`、`rqt_reconfigure`。

整机相关包不属于当前公开视觉仓库。串口规则绑定的是 USB 物理路径，迁移设备时须核对。机械臂节点可能把类别标签作为控制逻辑输入，因此纯视觉配置的正确标签不可未经联调直接替换整机运行配置。

新增的隔离冒烟测试只发布静态图片，使用单独 ROS master，没有启动串口、机械臂、相机或原自启动服务。
