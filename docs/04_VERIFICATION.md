# 实际验证记录与边界

本文记录已执行的验收及其历史证据名称。原始系统归档、主机取证和操作日志保存在开发者私有目录，未发布到 GitHub；下面的 `archives/`、`environment/`、`verification/` 路径是历史记录，不是复现所需下载项。公开仓库复现步骤见根目录 README 和 `repro/smoke_ros.py`。

本地记录日期：2026-10-04。Jetson 设备时钟显示 2025-08-08，因此下面日志内的旧日期是设备时间，不是另一次历史测试。

## 1. 已完成的验收

| 项目 | 结果 | 证据 |
|---|---|---|
| 原始归档远端/本地 SHA256 | 一致 | `archives/archive_info.json`、`archives/SHA256SUMS` |
| 原始文件/链接核对 | 31,964 条全部匹配，0 缺失、0 不一致 | `verification/integrity.json` |
| Windows 阅读副本 | 从原始归档展开 479 个文件，原始内容保持一致 | `verification/local_manifest.json` |
| 安装资料归档 SHA256 | 一致 | `archives/jetson-installers.sha256`、`archives/SHA256SUMS` |
| 验证结果归档 SHA256 | 一致 | `archives/jetson-validation.sha256`、`archives/SHA256SUMS` |
| 原源码独立编译 | 退出 0，`yolo_trt_node`、`usb_cam_node` 均成功构建 | `verification/build.log`、`build.exit` |
| 原部署 engine 反序列化及接口 | 两套通过，分别输出 `[1,7,2100]`、`[1,8,2100]` | `verification/engine-bindings.json` |
| 色环 ONNX→新 FP16 engine→trtexec 推理 | 退出 0，PASSED | `verification/rebuild-ring.log` |
| 物料 ONNX→新 FP16 engine→trtexec 推理 | 退出 0，PASSED | `verification/rebuild-wuliao.log` |
| 原部署色环/物料 engine 的 ROS 图片管线 | 两套均通过 | `verification/jetson-validation/smoke-original-*/result.json` |
| 新建色环/物料 engine 的 ROS 图片管线 | 两套均通过 | `verification/jetson-validation/smoke-rebuilt-*/result.json` |
| 新增纯视觉 launch 参数解析 | 通过，四类物料标签与阈值符合源码 | `verification/vision-launch-resolved.yaml` |
| 测试退出清理 | 测试 master 11321 无监听，原服务 active，原检测 PID 6275 仍在 | `verification/post-validation-state.txt` |

新建 engine 已随验证归档备份，并展开在：

```text
verification/jetson-validation/ring-rebuilt.engine
verification/jetson-validation/wuliao-rebuilt.engine
```

这些是本次的验证产物，原部署 engine 仍独立保存在 `source/ros_ws/src/bot_vision/engine_models/`。

## 2. ROS 测试方法

使用重新编译的原 C++ 节点、独立 ROS master `127.0.0.1:11321`，输入工程中已有的 `940.jpg`，统一 resize 到 640×480。没有接入相机、串口或机械臂。

`repro/smoke_ros.py` 检查模型成功加载、检测消息和 BGR8 可视化消息均收到、输出图像大小、类别合法、概率/坐标有限及宽高非负。每次测试的 `node.log`、`master.log`、`result.json`、`result.jpg` 均保存在 `verification/jetson-validation/smoke-*/`。

原/新 engine 对同一张图片的对比：

| 模型 | 检测标签集合 | 最大概率差 | 最大框坐标差 |
|---|---|---|---|
| 色环 | green_ring、blue_ring，两边一致 | 0.0027633 | 0.03125 像素 |
| 物料 | blue_wuliao，两边一致 | 0 | 0 像素 |

这说明重建模型能接入原 ROS 实现，且在该样例上结果接近；不是标注数据集准确率评估，也不证明所有输入完全等价。样例中的误检/漏检未按人工真值评判。

## 3. 验证中处理的问题

- 初版测试脚本连续运行时，端口预检查将 TCP `TIME_WAIT` 当作不可用；实测普通 bind 失败、`SO_REUSEADDR` bind 成功。已修正并连续重跑两套新 engine 通过，证据为 `verification/smoke-port-reproduction.txt`。原业务代码未为此修改。
- SSH 曾被断开。内核日志记录有线网卡 `eth0` 两次 Link Down/Up，重新连接后取回了完整色环测试结果，设备未重启。证据为 `verification/reconnection-diagnosis.txt`。
- `verification/smoke-original-ring.log` 是断连时产生的不完整客户端日志，其退出状态不能作为测试结论；以回收的 `jetson-validation/smoke-original-ring/result.json`、`node.log` 为准。
- 部分采集命令非零退出有明确范围：`service` 项因采集时 `/dev/video*` 和串口设备不存在；`model_search` 包含不存在/不可访问目录。完整执行状态见 `environment/capture_status.json`。

## 4. 没有验证或无法补齐的部分

1. **从零训练：**缺少准确 PT、训练集、原 YAML、超参数和评估记录。已验证路径从存档 ONNX/engine 开始。
2. **全新系统安装与完全离线恢复：**没有重刷 Jetson；保存了版本和部分安装资料，未打包全部系统 deb、全部 conda 包或整机镜像。
3. **其他硬件/软件版本：**未验证 Orin、Xavier、其他 GPU 或 TensorRT 10；engine 需在目标设备重建并验收。
4. **真实摄像头与整机联动：**采集时相机/串口设备节点不存在，没有验证实时视频、STM32、条码、机械臂、导航和完整自启动动作。
5. **精度和长期稳定性：**没有完整数据集评估或长时间压力测试。

本次没有修改 Jetson 的原源码、原模型、服务和配置，也没有保存登录密码。只在任务临时目录生成构建与测试文件。原系统负载会影响测试耗时，因此日志中的性能值只作为“确实执行过”的证据。

## 5. 在目标设备复现验收

按根目录 README 从源码编译、重建 engine，运行 `repro/smoke_ros.py` 并核对 `result.json`、`node.log`、`result.jpg`。两套模型分别验收。公开仓库不依赖开发者的系统归档校验工具，也不包含原训练集及精度评估基准。
