#ifndef IMAGEDETECTOR_H
#define IMAGEDETECTOR_H

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

struct FaceBox {
    float confidence;
    float x;
    float y; 
    float w;
    float h;
};

float IOUCalculate(const FaceBox& det_a, const FaceBox& det_b);
void NmsDetect(std::vector<FaceBox>& detections);

class ImageDetector : public TrtPipeline {
    public:
        ImageDetector(const std::string &onnxFile);
        ~ImageDetector();

    private:        
        int mInputH;
        int mInputW;

        virtual void _preprocessInput(std::shared_ptr<BufferManager> buffers, cv::Mat &img);
        virtual void _postprocessOutput(std::shared_ptr<BufferManager> buffers, cv::Mat &img);
};

#endif