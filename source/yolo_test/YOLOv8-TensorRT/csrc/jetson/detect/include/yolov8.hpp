#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace det;

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path, float conf_thresh = 0.25f);
    ~YOLOv8();

    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, const cv::Size& size);
    void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size);
    void infer();
    void postprocess(std::vector<Object>& objs);
    static void draw_objects(const cv::Mat& image,
                             cv::Mat& res,
                             const std::vector<Object>& objs,
                             const std::vector<std::string>& CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);

private:
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;

    std::vector<void*>   device_ptrs;
    std::vector<void*>   host_ptrs;
    std::vector<size_t>  bindingSizes;

    PreParam pparam;
    float    confThreshold;
    Logger   gLogger{nvinfer1::ILogger::Severity::kERROR};
};

// Implementation

YOLOv8::YOLOv8(const std::string& engine_file_path, float conf_thresh)
    : confThreshold(conf_thresh)
{
    // Load engine file
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    file.close();

    // Create runtime, engine, context, stream
    initLibNvInferPlugins(&gLogger, "");
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime);
    engine  = runtime->deserializeCudaEngine(buf.data(), size);
    assert(engine);
    context = engine->createExecutionContext();
    assert(context);
    cudaStreamCreate(&stream);

    // Gather binding info
    num_bindings = engine->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        Binding b;
        b.name  = engine->getBindingName(i);
        b.dsize = type_to_size(engine->getBindingDataType(i));
        if (engine->bindingIsInput(i)) {
            auto dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            b.dims = dims;
            b.size = get_size_by_dims(dims);
            input_bindings.push_back(b);
            num_inputs++;
            context->setBindingDimensions(i, dims);
        } else {
            auto dims = context->getBindingDimensions(i);
            b.dims = dims;
            b.size = get_size_by_dims(dims);
            output_bindings.push_back(b);
            num_outputs++;
        }
    }
}

YOLOv8::~YOLOv8()
{
    cudaStreamDestroy(stream);
    if (context) context->destroy();
    if (engine ) engine->destroy();
    if (runtime) runtime->destroy();
    for (auto p : device_ptrs) cudaFree(p);
    for (auto p : host_ptrs)   cudaFreeHost(p);
}

void YOLOv8::make_pipe(bool warmup)
{
    device_ptrs.clear();
    host_ptrs.clear();
    bindingSizes.clear();

    for (int i = 0; i < num_bindings; ++i) {
        auto dims = engine->getBindingDimensions(i);
        size_t elems = 1;
        for (int d = 0; d < dims.nbDims; ++d) elems *= dims.d[d];
        size_t byteSize = elems * type_to_size(engine->getBindingDataType(i));
        void* dev = nullptr;
        CHECK(cudaMalloc(&dev, byteSize));
        device_ptrs.push_back(dev);
        if (!engine->bindingIsInput(i)) {
            void* host = nullptr;
            CHECK(cudaHostAlloc(&host, byteSize, cudaHostAllocPortable));
            host_ptrs.push_back(host);
            bindingSizes.push_back(byteSize);
        }
    }

    if (warmup) {
        for (int k = 0; k < 10; ++k) {
            for (int i = 0; i < num_inputs; ++i) {
                auto& b = input_bindings[i];
                size_t bs = b.size * b.dsize;
                std::vector<uint8_t> zeros(bs);
                CHECK(cudaMemcpyAsync(device_ptrs[i],
                                      zeros.data(), bs,
                                      cudaMemcpyHostToDevice, stream));
            }
            infer();
        }
        std::cout << "[DEBUG] warmup done\n";
    }
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat nchw;
    auto& b = input_bindings[0];
    cv::Size sz(b.dims.d[3], b.dims.d[2]);
    letterbox(image, nchw, sz);
    context->setBindingDimensions(0, b.dims);
    CHECK(cudaMemcpyAsync(device_ptrs[0],
                          nchw.ptr<float>(),
                          nchw.total()*nchw.elemSize(),
                          cudaMemcpyHostToDevice,
                          stream));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, const cv::Size& size)
{
    cv::Mat nchw;
    letterbox(image, nchw, size);
    CHECK(cudaMemcpyAsync(device_ptrs[0],
                          nchw.ptr<float>(),
                          nchw.total()*nchw.elemSize(),
                          cudaMemcpyHostToDevice,
                          stream));
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size)
{
    float h0 = image.rows, w0 = image.cols;
    float nh = size.height, nw = size.width;
    float r  = std::min(nh/h0, nw/w0);
    int   nw_ = int(w0*r), nh_ = int(h0*r);
    cv::Mat tmp;
    cv::resize(image, tmp, {nw_, nh_});
    float dx = (nw - nw_)/2.f, dy = (nh - nh_)/2.f;
    cv::copyMakeBorder(tmp, tmp,
                       int(dy), int(nh-dy-nh_),
                       int(dx), int(nw-dx-nw_),
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    out.create({1,3,int(nh),int(nw)}, CV_32F);
    std::vector<cv::Mat> ch(3);
    cv::split(tmp, ch);
    int H = int(nh), W = int(nw);
    ch[2].convertTo(cv::Mat(H,W,CV_32F, out.ptr<float>()),    CV_32F, 1/255.0);
    ch[1].convertTo(cv::Mat(H,W,CV_32F, out.ptr<float>()+H*W),CV_32F, 1/255.0);
    ch[0].convertTo(cv::Mat(H,W,CV_32F, out.ptr<float>()+H*W*2),CV_32F, 1/255.0);
    pparam = {1/r, dx, dy, h0, w0};
}

void YOLOv8::infer()
{
    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; ++i) {
        CHECK(cudaMemcpyAsync(host_ptrs[i],
                              device_ptrs[num_inputs+i],
                              bindingSizes[i],
                              cudaMemcpyDeviceToHost,
                              stream));
    }
    CHECK(cudaStreamSynchronize(stream));
}

void YOLOv8::postprocess(std::vector<Object>& objs)
{
    auto dims = output_bindings[0].dims;    // [batch, attr, maxDet]
    int attr   = dims.d[1];
    int maxDet = dims.d[2];

    std::cout << "[DEBUG] output dims = ["
              << dims.d[0] << "," << attr << "," << maxDet << "]\n";

    float* dets = static_cast<float*>(host_ptrs[0]);
    int num_classes = attr - 4;    // 4 个坐标后面是每类概率

    // 临时收集：每个类别的所有检测
    std::vector<std::vector<Object>> byClass(num_classes);

    // 1) 扫描所有候选
    for (int i = 0; i < maxDet; ++i) {
        // 从网络输出读 (cx, cy, w, h)
        float cx = dets[0 * maxDet + i];
        float cy = dets[1 * maxDet + i];
        float bw = dets[2 * maxDet + i];
        float bh = dets[3 * maxDet + i];

        // 转成 (x0,y0)-(x1,y1) 相对于填充后图像
        float rx0 = cx - bw * 0.5f;
        float ry0 = cy - bh * 0.5f;
        float rx1 = cx + bw * 0.5f;
        float ry1 = cy + bh * 0.5f;

        // 调试打印
        std::cout << "[DEBUG] raw box["<<i<<"] = ("
                << rx0 << "," << ry0 << ") -> ("
                << rx1 << "," << ry1 << ")\n";

        // 找类别置信度
        float best_conf = 0.f;
        int   best_lbl  = 0;
        for (int c = 0; c < num_classes; ++c) {
            float sc = dets[(4 + c) * maxDet + i];
            if (sc > best_conf) {
                best_conf = sc;
                best_lbl  = c;
            }
        }
        if (best_conf <= confThreshold) continue;

        // 映射回原图
        float x0 = clamp((rx0 - pparam.dw) * pparam.ratio, 0.f, pparam.width);
        float y0 = clamp((ry0 - pparam.dh) * pparam.ratio, 0.f, pparam.height);
        float x1 = clamp((rx1 - pparam.dw) * pparam.ratio, 0.f, pparam.width);
        float y1 = clamp((ry1 - pparam.dh) * pparam.ratio, 0.f, pparam.height);

        // 正向化
        float lx = std::min(x0, x1), ty = std::min(y0, y1);
        float rx = std::max(x0, x1), by = std::max(y0, y1);

        Object o;
        o.rect.x      = lx;
        o.rect.y      = ty;
        o.rect.width  = rx - lx;
        o.rect.height = by - ty;
        o.prob        = best_conf;
        o.label       = best_lbl;
        byClass[best_lbl].push_back(o);
    }


    // 2) 对每个类别，只保留置信度最高的那个
    objs.clear();
    for (int c = 0; c < num_classes; ++c) {
        auto &v = byClass[c];
        if (v.empty()) continue;
        auto best_it = std::max_element(
            v.begin(), v.end(),
            [](auto &a, auto &b){ return a.prob < b.prob; }
        );
        objs.push_back(*best_it);

        // 打印最终保留的框
        const auto &o = *best_it;
        std::cout << "[DEBUG] keep class " << c
                  << " conf=" << o.prob
                  << " rect=(" << o.rect.x << "," << o.rect.y
                  << ")-(" << o.rect.x + o.rect.width
                  << "," << o.rect.y + o.rect.height << ")\n";
    }
}



void YOLOv8::draw_objects(const cv::Mat& image,
                          cv::Mat& res,
                          const std::vector<Object>& objs,
                          const std::vector<std::string>& CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (const auto& o : objs) {
        if (o.label < 0 || o.label >= (int)COLORS.size()) continue;

        // 先取两个角
        float x0 = o.rect.x;
        float y0 = o.rect.y;
        float x1 = o.rect.x + o.rect.width;
        float y1 = o.rect.y + o.rect.height;
        // 正向化：保证 x0<x1, y0<y1
        float lx = std::min(x0, x1), rx = std::max(x0, x1);
        float ty = std::min(y0, y1), by = std::max(y0, y1);
        std::cout << "[DEBUG] draw rect: ("
                  << lx << "," << ty << ") -> ("
                  << rx << "," << by << ")\n";

        // 构造一个合法的 Rect
        cv::Rect rect( int(lx), int(ty),
                       int(rx - lx), int(by - ty) );

        // 画框
        cv::Scalar c(
            COLORS[o.label][0],
            COLORS[o.label][1],
            COLORS[o.label][2]
        );
        cv::rectangle(res, rect, c, 2);

        // 画标签
        char buf[64];
        sprintf(buf, "%s %.1f%%",
                CLASS_NAMES[o.label].c_str(), o.prob * 100);
        cv::putText(res, buf,
                    cv::Point(int(lx), int(ty) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(255,255,255), 1);
    }
}

#endif // JETSON_DETECT_YOLOV8_HPP
