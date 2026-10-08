#!/bin/bash
source /opt/ros/melodic/setup.bash
source ~/bot_dev/bot_ws/devel/setup.bash
source ~/.bashrc   # 如果你把 ROS_IP 等写进了这里

# 启动 ROS 启动文件
roslaunch bot_vision yolo.launch

