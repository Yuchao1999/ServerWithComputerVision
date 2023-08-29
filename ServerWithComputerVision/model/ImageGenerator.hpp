#ifndef IMAGEGENERATOR_HPP
#define IMAGEGENERATOR_HPP

#include <iostream>
#include <string>
#include <queue>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "TrtPipeline.hpp"
#include "../server/utils.hpp"

class ImageGenerator : public TrtPipeline {
    public:
        ImageGenerator(const std::string &onnxFile, bool isDynamic);
        ~ImageGenerator();
    private:        
        virtual void _preprocessInput(std::shared_ptr<BufferManager> buffers, cv::Mat &img);
        virtual void _postprocessOutput(std::shared_ptr<BufferManager> buffers, cv::Mat &img);
};

#endif