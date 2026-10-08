#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <dynamic_reconfigure/server.h>
#include <bot_vision/UsbCamConfig.h>

cv::VideoCapture cap;
std::string fixed_device_path = "/dev/car_camera"; // 使用固定的设备符号链接
bot_vision::UsbCamConfig current_config; // 保存当前配置

// 应用摄像头参数的函数
void applyCameraSettings() {
    if (!cap.isOpened()) return;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, current_config.frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, current_config.frame_height);
    cap.set(cv::CAP_PROP_FPS, current_config.publish_rate);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, current_config.auto_exposure ? 3 : 1);
    if (!current_config.auto_exposure) {
        cap.set(cv::CAP_PROP_EXPOSURE, 0.1); // Manual exposure
    }
}

// 动态重配置回调
void callback(bot_vision::UsbCamConfig &config, uint32_t level) {
    current_config = config; // 保存最新配置
    applyCameraSettings();   // 立即应用
    ROS_INFO("[Reconfigure] width=%d height=%d fps=%.1f auto_exp=%s",
        config.frame_width, config.frame_height, config.publish_rate,
        config.auto_exposure ? "true" : "false");
}

// 尝试打开摄像头
bool openCamera() {
    cap.open(fixed_device_path, cv::CAP_V4L2);
    if (cap.isOpened()) {
        applyCameraSettings();
        ROS_INFO("Opened camera at %s", fixed_device_path.c_str());
        return true;
    }
    return false;
}

// 尝试重新打开摄像头
bool reconnectCamera() {
    cap.release(); // 释放旧资源
    ros::Duration(0.5).sleep(); // 短暂等待避免高频重试
    
    // 尝试打开固定设备路径
    if (openCamera()) {
        ROS_WARN("Camera reconnected at %s", fixed_device_path.c_str());
        return true;
    }
    
    return false;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "usb_cam_node");
    ros::NodeHandle nh("~");
    
    // 初始参数
    std::string image_topic;
    nh.param("image_topic", image_topic, std::string("/camera/image_raw"));
    
    // 初始配置（与.cfg默认值一致）
    current_config.frame_width = 640;
    current_config.frame_height = 480;
    current_config.publish_rate = 30.0;
    current_config.auto_exposure = true;
    
    // 首次打开摄像头
    if (!openCamera()) {
        ROS_ERROR("Failed to open camera at %s", fixed_device_path.c_str());
        return 1;
    }
    
    // 动态重配置
    dynamic_reconfigure::Server<bot_vision::UsbCamConfig> server;
    server.setCallback(boost::bind(&callback, _1, _2));
    
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise(image_topic, 1);
    
    ros::Rate loop_rate(30.0);
    ROS_INFO("usb_cam_node started with auto-reconnect support (fixed device: %s).", fixed_device_path.c_str());
    
    int fail_count = 0;
    const int MAX_FAILS = 30; // 约1秒（30Hz循环）
    bool reconnect_in_progress = false;
    
    while (ros::ok()) {
        // 检查摄像头状态
        if (!cap.isOpened() || fail_count > MAX_FAILS) {
            if (!reconnect_in_progress) {
                ROS_WARN("Attempting to reconnect to camera...");
                reconnect_in_progress = true;
            }
            
            if (reconnectCamera()) {
                ROS_WARN("Camera reconnection successful!");
                fail_count = 0;
                reconnect_in_progress = false;
            } else {
                ROS_ERROR_THROTTLE(2.0, "Waiting for camera to reconnect...");
                fail_count = 0; // 重置计数避免连续触发
                ros::Duration(0.5).sleep(); // 重连失败时等待
                ros::spinOnce();
                continue;
            }
        }
        
        // 尝试读取帧
        cv::Mat frame;
        if (!cap.read(frame)) {
            fail_count++;
            if (fail_count % 10 == 0) { // 每10次失败记录一次
                ROS_WARN_THROTTLE(1.0, "Failed to capture frame (%d attempts)", fail_count);
            }
            ros::spinOnce();
            continue;
        }
        
        // 成功读取帧
        fail_count = 0;
        reconnect_in_progress = false;
        
        // 发布图像
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
            std_msgs::Header(), "bgr8", frame).toImageMsg();
        pub.publish(msg);
        
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    cap.release();
    return 0;
}