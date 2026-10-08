#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/Trigger.h>
#include <XmlRpcValue.h>
#include <opencv2/opencv.hpp>
#include <dynamic_reconfigure/server.h>
#include <bot_vision/YoloTRTConfig.h>
#include <boost/filesystem.hpp> // 添加文件系统支持

#include "bot_vision/common.hpp"
#include "bot_vision/yolov8.hpp"
#include "bot_vision/Detection.h"
#include "bot_vision/DetectionArray.h"

#include <chrono>
#include <memory>
#include <random>
#include <sstream>
#include <algorithm>

namespace fs = boost::filesystem; // 文件系统命名空间

class YOLOv8Node {
public:
  YOLOv8Node(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), it_(nh_)
  {
    target_size_ = cv::Size(320, 320);
    visualize_ = true;
    model_loaded_ = false; // 初始模型未加载

    if (!loadParams()) {
      ROS_ERROR("Failed to load parameters, but node will continue running");
    }

    // 尝试初始化检测器，但不强制退出
    initDetector();

    image_sub_      = it_.subscribe("camera/image_raw", 1,
                          &YOLOv8Node::imageCallback, this);
    detections_pub_ = nh_.advertise<bot_vision::DetectionArray>("detections", 10);
    result_pub_     = it_.advertise("detection_result", 1);
    reload_srv_     = pnh_.advertiseService("reload_model",
                          &YOLOv8Node::reloadCallback, this);

    dyn_server_.setCallback(
      boost::bind(&YOLOv8Node::dynCallback, this, _1, _2));

    ROS_INFO("YOLOv8Node ready with dynamic reconfigure support.");
  }

private:
  bool loadParams() {
    if (!pnh_.getParam("engine_path", engine_path_) || engine_path_.empty()) {
      ROS_FATAL("~engine_path missing");
      return false;
    }

    // 提取模型名称用于显示
    model_name_ = extractModelName(engine_path_);

    // 加载类别名称字符串
    std::string class_names_str;
    if (!pnh_.getParam("class_names_str", class_names_str) || class_names_str.empty()) {
      ROS_FATAL("~class_names_str missing or empty");
      return false;
    }

    // 分割逗号分隔的类别名称
    class_names_.clear();
    std::istringstream ss(class_names_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      // 去除首尾空格
      token.erase(0, token.find_first_not_of(" "));
      token.erase(token.find_last_not_of(" ") + 1);
      if (!token.empty()) {
        class_names_.push_back(token);
      }
    }

    if (class_names_.empty()) {
      ROS_FATAL("No valid class names found in class_names_str");
      return false;
    }

    generateColorPalette(class_names_.size());

    ROS_INFO("Loaded params: engine=%s, classes=%zu", engine_path_.c_str(), class_names_.size());
    for (size_t i = 0; i < class_names_.size(); ++i) {
      ROS_INFO("  Class %zu: '%s'", i, class_names_[i].c_str());
    }
    return true;
  }

  // 从路径中提取模型名称
  std::string extractModelName(const std::string& path) {
    if (path.empty()) return "Unknown Model";
    
    fs::path p(path);
    std::string name = p.filename().string();
    
    // 移除扩展名
    size_t pos = name.find_last_of(".");
    if (pos != std::string::npos) {
      name = name.substr(0, pos);
    }
    
    return name;
  }

  void generateColorPalette(size_t n) {
    // 静态预定义颜色列表
    static const std::vector<std::vector<unsigned int>> predefined_colors = {
        {0, 0, 128},     // 红色 (BGR)
        {0, 128, 0},     // 绿色
        {128, 0, 0},     // 蓝色
        {128, 0, 128},   // 紫色
        {255, 0, 255},   // 品红
        {255, 255, 0},   // 青色
        {0, 128, 128},   // 橄榄色
        {128, 128, 0},   // 深绿色
        {0, 0, 128},     // 海军蓝
        {128, 0, 0},     // 栗色
        {255, 0, 255},   // 深紫色
        {0, 128, 128}    // 蓝绿色
    };

    colors_.clear();
    
    // 首先使用预定义颜色
    for (size_t i = 0; i < n && i < predefined_colors.size(); ++i) {
        colors_.push_back(predefined_colors[i]);
    }

    // 如果需要更多颜色，生成随机颜色
    if (colors_.size() < n) {
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> dist(0, 255);
        while (colors_.size() < n) {
            colors_.emplace_back(std::vector<unsigned int>{
                static_cast<unsigned int>(dist(rng)),
                static_cast<unsigned int>(dist(rng)),
                static_cast<unsigned int>(dist(rng))
            });
        }
    }
    
    ROS_INFO("Generated color palette with %zu colors", colors_.size());
  }

  bool initDetector() {
    // 检查模型文件是否存在
    if (!fs::exists(engine_path_)) {
      ROS_ERROR("Model file not found: %s", engine_path_.c_str());
      model_loaded_ = false;
      model_name_ = "Unknown Model"; // 更新模型名称为未知
      return false;
    }

    try {
      detector_.reset(new YOLOv8(engine_path_, conf_thresh_));
      detector_->make_pipe(true);
      model_loaded_ = true;
      model_name_ = extractModelName(engine_path_); // 更新模型名称
      ROS_INFO("Model loaded successfully: %s", model_name_.c_str());
      return true;
    } catch (const std::exception& e) {
      ROS_ERROR("YOLOv8 init failed: %s", e.what());
      model_loaded_ = false;
      model_name_ = "Unknown Model"; // 更新模型名称为未知
      return false;
    }
  }

  void dynCallback(bot_vision::YoloTRTConfig &config, uint32_t level) {
    // 保存旧值以便出错时恢复
    std::string old_engine = engine_path_;
    float old_conf = conf_thresh_;
    std::vector<std::string> old_classes = class_names_;
    
    // 更新参数
    conf_thresh_ = static_cast<float>(config.conf_thresh);
    engine_path_ = config.engine_path;
    
    // 更新类别名称字符串并重新解析
    std::string class_names_str = config.class_names_str;
    
    // 分割逗号分隔的类别名称
    class_names_.clear();
    std::istringstream ss(class_names_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      token.erase(0, token.find_first_not_of(" "));
      token.erase(token.find_last_not_of(" ") + 1);
      if (!token.empty()) {
        class_names_.push_back(token);
      }
    }
    
    if (class_names_.empty()) {
      ROS_ERROR("No valid class names found in class_names_str, reverting");
      class_names_ = old_classes;
      return;
    }
    
    // 重新生成颜色调色板
    generateColorPalette(class_names_.size());
    
    // 尝试重新初始化检测器
    if (!initDetector()) {
      ROS_ERROR("Failed to reload detector with new config, reverting");
      engine_path_ = old_engine;
      conf_thresh_ = old_conf;
      class_names_ = old_classes;
      generateColorPalette(class_names_.size()); // 恢复颜色调色板
    } else {
      ROS_INFO("Updated conf_thresh=%.2f, engine=%s, classes=%zu",
               conf_thresh_, engine_path_.c_str(), class_names_.size());
    }
  }

  bool reloadCallback(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    auto old_engine = engine_path_;
    auto old_classes = class_names_;
    
    if (!loadParams() || !initDetector()) {
      engine_path_ = old_engine;
      class_names_ = old_classes;
      res.success = false;
      res.message = "reload failed";
    } else {
      res.success = true;
      res.message = "reload ok";
    }
    return true;
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv::Mat image;
    try {
      image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (...) {
      ROS_ERROR("cv_bridge error");
      return;
    }

    cv::Mat vis = image.clone();
    int W = vis.cols, H = vis.rows;

    // 模型未加载时的处理
    if (!model_loaded_) {
      // 显示错误信息
      cv::putText(vis, "MODEL NOT LOADED", 
                  cv::Point(W/2-100, H/2), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0,
                  cv::Scalar(0, 0, 255), 2);
                  
      cv::putText(vis, "Check engine path: " + engine_path_, 
                  cv::Point(10, 30), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 255), 1);
      
      // 发布空检测结果
      bot_vision::DetectionArray da;
      da.header = msg->header;
      detections_pub_.publish(da);
      
      // 发布可视化结果
      result_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg());
      return;
    }

    // 模型已加载，正常处理
    detector_->copy_from_Mat(image, target_size_);
    auto t0 = std::chrono::high_resolution_clock::now();
    detector_->infer();
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<det::Object> objs;
    detector_->postprocess(objs);

    bot_vision::DetectionArray da;
    da.header = msg->header;
    
    // 添加类别索引安全检查
    for (auto& o : objs) {
      // 检查类别索引是否在有效范围内
      if (o.label < 0 || o.label >= static_cast<int>(class_names_.size())) {
        ROS_WARN("Detected invalid class index: %d (max: %zu), skipping...", 
                 o.label, class_names_.size() - 1);
        continue;
      }
      
      bot_vision::Detection d;
      d.label = class_names_[o.label];
      d.prob = o.prob;
      d.x = o.rect.x; 
      d.y = o.rect.y; 
      d.w = o.rect.width; 
      d.h = o.rect.height;
      da.detections.push_back(d);
    }
    detections_pub_.publish(da);

    if (visualize_) {
      for (auto& o : objs) {
        // 双重检查类别索引有效性
        if (o.label < 0 || o.label >= static_cast<int>(colors_.size())) {
          ROS_WARN_ONCE("Visualization skipped for invalid class index: %d", o.label);
          continue;
        }
        
        auto& v = colors_[o.label];
        cv::Scalar col(v[0], v[1], v[2]);
        cv::rectangle(vis, o.rect, col, 2);
        int cx = int(o.rect.x + o.rect.width * 0.5f);
        int cy = int(o.rect.y + o.rect.height * 0.5f);
        cv::line(vis, cv::Point(cx - 5, cy), cv::Point(cx + 5, cy), col, 2);
        cv::line(vis, cv::Point(cx, cy - 5), cv::Point(cx, cy + 5), col, 2);
      }

      int cx = W / 2, cy = H / 2;
      cv::Scalar yellow(0, 255, 255);
      cv::line(vis, cv::Point(cx - 10, cy), cv::Point(cx + 10, cy), yellow, 2);
      cv::line(vis, cv::Point(cx, cy - 10), cv::Point(cx, cy + 10), yellow, 2);

      double ms = std::chrono::duration_cast<
          std::chrono::microseconds>(t1 - t0).count() / 1000.0;
      double fps = ms > 0 ? 1000.0 / ms : 0.0;
      int lines = 2 + objs.size(); // 增加一行用于显示模型名称
      int ph = lines * 20, pw = 300;
      cv::Mat roi = vis(cv::Rect(0, 0, pw, ph));
      cv::Mat bg(ph, pw, CV_8UC3, cv::Scalar(50, 50, 50));
      cv::addWeighted(bg, 0.6, roi, 0.4, 0, roi);

      int y = 16;
      
      // 显示模型名称
      cv::putText(vis, "Model: " + model_name_,
                  cv::Point(5, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 255, 255), 1);
      y += 20;
      
      // 显示FPS
      cv::putText(vis, cv::format("FPS: %.1f", fps),
                  cv::Point(5, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 255, 255), 1);
      
      // 添加类别名称显示的安全检查
      for (size_t i = 0; i < objs.size(); ++i) {
        y += 20;
        auto& o = objs[i];
        
        // 再次检查类别索引
        if (o.label < 0 || o.label >= static_cast<int>(class_names_.size())) {
          continue;
        }
        
        cv::putText(vis,
                    cv::format("%s %.1f%% (%d,%d)",
                               class_names_[o.label].c_str(),
                               o.prob * 100,
                               int(o.rect.x + o.rect.width * 0.5f),
                               int(o.rect.y + o.rect.height * 0.5f)),
                    cv::Point(5, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
      }

      result_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg());
    }
  }

  ros::NodeHandle nh_, pnh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Publisher detections_pub_;
  image_transport::Publisher result_pub_;
  ros::ServiceServer reload_srv_;

  std::unique_ptr<YOLOv8> detector_;
  std::string engine_path_;
  std::vector<std::string> class_names_;
  std::vector<std::vector<unsigned int>> colors_;
  cv::Size target_size_;
  float conf_thresh_ = 0.3f;
  bool visualize_;
  bool model_loaded_; // 模型加载状态标志
  std::string model_name_; // 模型名称

  dynamic_reconfigure::Server<bot_vision::YoloTRTConfig> dyn_server_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "yolo_trt_node");
  ros::NodeHandle nh, pnh("~");
  cudaSetDevice(0);
  YOLOv8Node node(nh, pnh);
  ros::spin();
  return 0;
}