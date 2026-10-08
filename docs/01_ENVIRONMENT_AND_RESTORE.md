# 环境与部署

本指南以检出的仓库为输入。按根目录 [README](../README.md) 完成编译、模型转换、运行和无相机验收；下面说明环境选择与部署边界。

## 已验证的 SDK

| 组件 | 版本 |
| --- | --- |
| 硬件 | Jetson TX2 NX / ARM64 / 4 GB 共享内存 |
| 系统 | Ubuntu 18.04.6 / L4T 32.7.6 / 内核 4.9.337-tegra |
| CUDA | 10.2.300 |
| TensorRT | C++ 8.2.1-1+cuda10.2 / Python 8.2.1.9 |
| cuDNN | 8.2.1.32 |
| OpenCV | 4.1.1 |
| ROS | Melodic 1.14.13 |
| GCC / C++ | 7.5.0 / C++14 |
| CMake | 验证使用 3.22.6 |
| Python | ROS 构建/测试使用系统 Python 2.7 |

C++ 节点不加载 PyTorch/Ultralytics；部署依赖是 ROS、OpenCV、CUDA、TensorRT。不要为了运行 C++ 程序误装完整训练环境。

先安装匹配 TX2 NX 的 NVIDIA SDK 和 ROS Melodic。系统、NVIDIA SDK、旧 ROS 软件源与完整离线系统安装包不由本仓库提供。旧版环境软件源是否仍可获取须在目标设备确认；不要直接升级最新版 CUDA/TensorRT 来尝试加载旧 engine。

核对环境：

```bash
uname -m
cat /etc/nv_tegra_release
nvcc --version
dpkg-query -W 'libnvinfer*' 'libcudnn*'
source /opt/ros/melodic/setup.bash
rosversion -d
cmake --version
```

## 构建工作空间

从仓库中的 `source/ros_ws/src/bot_vision` 复制视觉包到新 catkin 工作空间即可，不需要原始系统归档。具体依赖安装与命令见 README。

4 GB 设备推荐 `catkin_make -j1 -l2`。catkin 的 Python 使用 `/usr/bin/python2`，不要将其无条件改成训练环境的 Python。工程使用 C++14；CMake 显式查找 ARM64 的 TensorRT 头文件和库，移植到其他架构需要调整。

使用 `devel/setup.bash`，因为历史 `install(FILES Detection.msg DetectionArray.msg ...)` 的源路径不正确，不保证 `catkin_make install` 可用。没有执行全新刷机或离线系统恢复验收。

## 模型与运行

GPU 或 TensorRT 环境不同，先从匹配 ONNX 重建 engine，见 [模型转换](02_MODELS_AND_CONVERSION.md)。

`repro/vision_only.launch` 默认使用物料四类模型，只启动检测节点，默认不启动相机。输入话题为 `/camera/image_raw`，支持 `image_topic` remap。输出是 `/detections` 与 `/detection_result`。

相机节点固定打开 `/dev/car_camera`。更换相机后，先核对属性与权限，再按需调整 `config/udev/99-car-camera.rules`。原 `cam_index` 参数不生效。运行图像尺寸建议至少 640×480，避免可视化面板超出图像。

测试建议使用独立 ROS master `127.0.0.1:11321`。终端之间要使用同一 `ROS_MASTER_URI`、工作空间和主机设置；不要终止无关 master 或重复启动同名检测节点。

## 无硬件验收

`repro/smoke_ros.py` 会启动自己的 master、检测节点和静态图片发布，检查模型加载、检测消息、标注图像、图像尺寸、类别与数值合法性；结束后清理自己启动的进程。

运行前关闭占用 11321 的测试 launch，并使用新的输出目录。命令及结果判断见 README。该方法不需要相机、串口或机械臂，也不能代替真实相机吞吐、完整数据集精度与整机联动测试。

## 历史整机配置

`config/systemd/`、`config/start_ros_yolo.sh` 与原 `yolo.launch` 是历史整机配置，含固定部署路径，并会启动串口、机械臂、条码识别和 GUI。纯视觉复现不要启用这些服务。整机迁移需要重新确认所有设备、用户、路径、权限、网络及类别与控制逻辑的对应关系。
