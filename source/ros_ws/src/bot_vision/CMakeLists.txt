cmake_minimum_required(VERSION 3.1.0)
project(bot_vision)

# use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#######################
## Find catkin macros
find_package(catkin REQUIRED COMPONENTS
  std_srvs
  roscpp
  image_transport
  cv_bridge
  sensor_msgs
  std_msgs
  message_generation
  dynamic_reconfigure
)

add_service_files(
  FILES
  GetTaskCode.srv
)

## ROS messages
add_message_files(
  FILES
    Detection.msg
    DetectionArray.msg
)
generate_messages(
  DEPENDENCIES std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/UsbCam.cfg
  cfg/YoloTRT.cfg
  cfg/STMTaskConfig.cfg
)

## Find system dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA   REQUIRED)
find_package(dynamic_reconfigure REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIB           nvinfer        HINTS /usr/lib/aarch64-linux-gnu)
find_library(TENSORRT_PLUGIN_LIB    nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES bot_vision
  CATKIN_DEPENDS roscpp image_transport cv_bridge sensor_msgs std_msgs message_runtime
  DEPENDS OpenCV CUDA
)

###########
## Build ##
###########

# include paths
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

# link directories (if needed)
link_directories(
  ${CUDA_LIBRARY_DIRS}
)

###########
## Nodes ##
###########

set(YOLO_SRCS
  src/yolo_trt_node.cpp
)

add_executable(yolo_trt_node ${YOLO_SRCS})
# USB camera node (if still needed)
add_executable(usb_cam_node src/usb_cam_node.cpp)

add_dependencies(yolo_trt_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(usb_cam_node ${PROJECT_NAME}_gencfg)
add_dependencies(yolo_trt_node ${PROJECT_NAME}_gencfg)

target_link_libraries(yolo_trt_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
  ${TENSORRT_LIB}
  ${TENSORRT_PLUGIN_LIB}
  nvinfer
  nvinfer_plugin
)

target_link_libraries(usb_cam_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(TARGETS yolo_trt_node usb_cam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  Detection.msg
  DetectionArray.msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
