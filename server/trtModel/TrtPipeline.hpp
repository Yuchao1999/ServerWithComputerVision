#ifndef TRTPIPELINE_H
#define TRTPIPELINE_H

#include <iostream>
#include <vector>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>
#include "utils.hpp"
#include "buffers.hpp"

class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class TrtPipeline {
    public:
        TrtPipeline(const std::string onnxFile, bool isDynamic = false);
        ~TrtPipeline();
        void inference(cv::Mat &image, 
            std::shared_ptr<BufferManager> buffers, 
            std::shared_ptr<nvinfer1::IExecutionContext> context);
        std::shared_ptr<BufferManager> createBuffer();
        std::shared_ptr<BufferManager> createBuffer(std::shared_ptr<nvinfer1::IExecutionContext> context);
        std::shared_ptr<nvinfer1::IExecutionContext> createContext();
        std::shared_ptr<nvinfer1::IExecutionContext> createContext(cv::Size size);
    private:
        void loadTrtModel();
        void loadOnnxModel();
    protected:
        Logger gLogger;
        std::string _onnxModelFile;
        std::string _trtModelFile;

        std::shared_ptr<nvinfer1::IRuntime> mRuntime;
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

        bool _isDynamic;

        // image preprocess function
        virtual void _preprocessInput(std::shared_ptr<BufferManager> buffers, cv::Mat &img) = 0;

        // image postprocess function
        virtual void _postprocessOutput(std::shared_ptr<BufferManager> buffers, cv::Mat &img) = 0;
};

#endif
