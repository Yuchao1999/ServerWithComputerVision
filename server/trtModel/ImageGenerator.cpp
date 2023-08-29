
#include "ImageGenerator.hpp"

const std::string INPUT_NAME =  "animeganv3_input:0";
const std::string OUTPUT_NAME =  "generator/main/out_layer:0";

using namespace nvinfer1;

ImageGenerator::ImageGenerator(const std::string &onnxFile, bool isDynamic) 
    : TrtPipeline(onnxFile, isDynamic) {
}

ImageGenerator::~ImageGenerator() {

}

void ImageGenerator::_preprocessInput(std::shared_ptr<BufferManager> buffers, cv::Mat &img) {
    float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer(INPUT_NAME));
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_32FC3, hostDataBuffer);
    img.convertTo(dst_img, CV_32FC3);
}

void ImageGenerator::_postprocessOutput(std::shared_ptr<BufferManager> buffers, cv::Mat &img) {
    float* outputBuffer = static_cast<float*>(buffers->getHostBuffer(OUTPUT_NAME));
    cv::Mat image(img.rows, img.cols, CV_32FC3, outputBuffer);
    cv::normalize(image, img, 0, 255, cv::NORM_MINMAX, CV_8UC3);
}
