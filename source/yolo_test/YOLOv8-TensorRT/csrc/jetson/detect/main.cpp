//
// Created by ubuntu on 3/16/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "red_ring",         "green_ring",    "blue_ring"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 0, 255},   {0, 255, 0},   {255, 0, 0}
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const fs::path    path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    auto yolov8 = new YOLOv8(engine_file_path, 0.8f);
    yolov8->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                 || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat             res, image;
    cv::Size            size = cv::Size{320, 320};
    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}