#pragma once

#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <glog/logging.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "buffers.hpp"
#include "timer.hpp"
#include "misc.hpp"

namespace MODEL {
using namespace std;
using namespace nvinfer1;
using samplesCommon::BufferManager;
template <typename T>
using RtUniquePtr = std::unique_ptr<T, misc::InferDeleter>;

const size_t MAX_TRT_SIZE = 1 << 28;
const size_t MAX_WORKSPACE = 8 * (1UL << 30);
const bool ENABLE_TRT_CACHE = true;

class Logger : public ILogger {
    void log(Severity severity, const char *msg) override {
        LOG(INFO) << msg;
    }
} gLogger;

struct RT_data {
    ICudaEngine *engine;
    IExecutionContext *ctx;
    BufferManager *buffers;
    Dims inpDim;
    string inpName;
    int bindingInputIndex;
    vector<Dims> outDims;
    vector<string> outNames;
    vector<int> bindingOutputIndexes;
};

class Model {
   public:
    bool init(const char *);
    bool setInpData(shared_ptr<float[]>);
    bool preprocess(const cv::Mat& img, float* buf);
    bool infer();

    int batch_size(){return _rt_data.inpDim.d[0];};
    int input_height(){return _rt_data.inpDim.d[2];};
    int input_width(){return _rt_data.inpDim.d[3];};

   private:
    bool _build();
    void _warmup();

   private:
    RT_data _rt_data;
    string _onnx_file;
};

bool Model::init(const char* onnx_file) {
    LOG(INFO) << "start init: " << onnx_file;
    _onnx_file = onnx_file;
    if(!this->_build()){
        LOG(ERROR) << "build fails";
        return false;
    }
    this->_warmup();
    LOG(INFO) << "init finished";
    return true;
}

bool Model::_build(){
    string trt_file = _onnx_file.substr(0, _onnx_file.size() - 5);
    trt_file.insert(trt_file.size(), ".trt");
    if (ENABLE_TRT_CACHE && cv::utils::fs::exists(trt_file.c_str())) {
        LOG(INFO) << "trt cache exists: " << trt_file;
        IRuntime* runtime = createInferRuntime(gLogger);
        std::fstream trt_stream(trt_file.c_str(), std::ios::binary | std::ios::in);
        if (!trt_stream.is_open()) {
            LOG(INFO) << "Opening trt file fails.";
            return false;
        }
        shared_ptr<char> modelData(new char[MAX_TRT_SIZE]);
        trt_stream.read(modelData.get(), MAX_TRT_SIZE);
        size_t modelSize = trt_stream.gcount();
        ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.get(), modelSize, nullptr);
        if (!engine) {
            LOG(INFO) << "deserializeCudaEngine fails.";
            return false;
        }
        _rt_data.engine = engine;
    } else {
        if (!ENABLE_TRT_CACHE) {
            LOG(INFO) << "ENABLE_TRT_CACHE is false";
        } else {
            LOG(INFO) << "trt cache not exists: " << trt_file;
        }
        auto builder = RtUniquePtr<IBuilder>(createInferBuilder(gLogger));
        if (!builder) {
            LOG(ERROR) << "createInferBuilder fails";
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = RtUniquePtr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            LOG(ERROR) << "createNetworkV2 fails";
            return false;
        }
        
        auto parser = RtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            LOG(ERROR) << "createParser fails";
            return false;
        }

        auto parsed = parser->parseFromFile(_onnx_file.c_str(), (int)ILogger::Severity::kINFO);
        if (!parsed) {
            LOG(ERROR) << "parseFromFile fails";
            return false;
        }
        

        auto config = RtUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            LOG(ERROR) << "createBuilderConfig";
            return false;
        }
        config->setMaxWorkspaceSize(MAX_WORKSPACE);
        if(builder->platformHasFastFp16()){
            LOG(INFO) << "Using fp16";
            config->setFlag(BuilderFlag::kFP16);
        }

        // if(builder->getNbDLACores() > 0){
        //     LOG(INFO) << "Enable DLA";
        //     config->setFlag(BuilderFlag::kGPU_FALLBACK);
        //     config->setDefaultDeviceType(DeviceType::kDLA);
        //     config->setDLACore(0);
        // }
        
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        if (!engine) {
            LOG(ERROR) << "buildEngineWithConfig fails";
            return false;
        }
        _rt_data.engine = engine;
        IHostMemory* serializedEngine = engine->serialize();
        if (serializedEngine == nullptr) {
            LOG(ERROR) << "Engine serialization failed" << std::endl;
            return false;
        }
        std::ofstream engineFile(trt_file, std::ios::binary);
        engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
        serializedEngine->destroy();
    }

    int nbBindings = _rt_data.engine->getNbBindings();
    for (int i = 0; i < nbBindings; i++) {
        if (_rt_data.engine->bindingIsInput(i)) {
            _rt_data.bindingInputIndex = i;
            _rt_data.inpName = _rt_data.engine->getBindingName(i);
            _rt_data.inpDim = _rt_data.engine->getBindingDimensions(i);
        } else {
            _rt_data.bindingOutputIndexes.push_back(i);
            _rt_data.outNames.push_back(_rt_data.engine->getBindingName(i));
            _rt_data.outDims.push_back(_rt_data.engine->getBindingDimensions(i));
        }
    }
    _rt_data.ctx = _rt_data.engine->createExecutionContext();
    auto inp_binding = _rt_data.engine->getBindingIndex(_rt_data.inpName.c_str());
    _rt_data.ctx->setBindingDimensions(inp_binding, _rt_data.inpDim);
    if (!_rt_data.ctx) {
        LOG(ERROR) << "createExecutionContext fails";
        return false;
    }
    _rt_data.buffers = new BufferManager(_rt_data.engine);
    return true;
}

bool Model::preprocess(const cv::Mat& img, float* buf) {
    if(!buf){
        LOG(ERROR) << "out buf is NULL.";
        return false;
    }
    const int inputH = _rt_data.inpDim.d[2];
    const int inputW = _rt_data.inpDim.d[3];
    if (img.rows != inputH || img.cols != inputW) {
        LOG(ERROR) << "img.rows = " << img.rows << "; "
                   << "net inputH = " << inputH << ";";
        LOG(ERROR) << "img.cols = " << img.cols << "; "
                   << "net inputW = " << inputW << ";";
        return false;
    }
    const size_t _s0 = inputH * inputW;
    for (int r = 0; r < img.rows; r++) {
        const uint8_t * ptr = img.ptr(r);
        for (int c = 0; c < img.cols; c++) {
            uint32_t v = ((uint32_t *)(ptr + c * 3))[0];
            uint8_t b_v = 0x000000FF & (v);
            uint8_t g_v = 0x000000FF & (v >> 8);
            uint8_t r_v = 0x000000FF & (v >> 16);
            size_t _s1 = r * inputW;
            buf[_s1 + c] = r_v;
            buf[_s0 + _s1 + c] = g_v;
            buf[_s0 + _s0 + _s1 + c] = b_v;
        }
    }
    return true;
}

bool Model::setInpData(shared_ptr<float[]> inp_buf){
    float* hostDataBuffer = static_cast<float*>(_rt_data.buffers->getHostBuffer(_rt_data.inpName));
    size_t buffer_size = _rt_data.buffers->size(_rt_data.inpName);
    memcpy(hostDataBuffer, inp_buf.get(), buffer_size);
    return true;
}

void Model::_warmup(){
    LOG(INFO) << ">>> start warmup";
    auto d = _rt_data.inpDim.d;
    size_t vol = d[0]*d[1]*d[2]*d[3];
    unique_ptr<float []> fake_inp(new float[vol]);
    for(int i=0; i<10; i++){
        for(int j=0; j<vol; j++){
            fake_inp[j] = std::rand() % 255;
        }
        float* hostDataBuffer = static_cast<float*>(_rt_data.buffers->getHostBuffer(_rt_data.inpName));
        size_t buffer_size = _rt_data.buffers->size(_rt_data.inpName);
        memcpy(hostDataBuffer, fake_inp.get(), sizeof(float) * vol);
        
        _rt_data.buffers->copyInputToDevice();
        bool status = _rt_data.ctx->executeV2(_rt_data.buffers->getDeviceBindings().data());
        if (!status){
            throw runtime_error("warmup execution fails.");
        }
        _rt_data.buffers->copyOutputToHost();

    }
    LOG(INFO) << ">>> warmup finished.";
}

TIMER::Timer timer_execute("det::execute");
TIMER::Timer timer_buffer_h2d("det::buffer_h2d");
TIMER::Timer timer_buffer_d2h("det::buffer_d2h");
bool Model::infer() {
    auto buffers = _rt_data.buffers;
    timer_buffer_h2d.start();
    buffers->copyInputToDevice();
    timer_buffer_h2d.pause();

    timer_execute.start();
    bool status = _rt_data.ctx->executeV2(buffers->getDeviceBindings().data());
    if (!status) {
        return false;
    }
    timer_execute.pause();

    timer_buffer_d2h.start();
    buffers->copyOutputToHost();
    timer_buffer_d2h.pause();
    return true;
}


}  // namespace MODEL
