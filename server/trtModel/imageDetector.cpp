#include "imageDetector.hpp"

const std::string INPUT_NAME =  "input.1";
const std::string OUTPUT_HEATMAP =  "537";
const std::string OUTPUT_SCALE =  "538";
const std::string OUTPUT_OFFSET =  "539";
const std::string OUTPUT_LANDMARKS =  "540";

const float confThreash = 0.5;
const float NMSThreash = 0.2;

using namespace nvinfer1;

float IOUCalculate(const FaceBox& det_a, const FaceBox& det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
        std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

void NmsDetect(std::vector<FaceBox>& detections) {
    sort(detections.begin(), detections.end(), [=](const FaceBox& left, const FaceBox& right) {
        return left.confidence > right.confidence;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i], detections[j]);
            if (iou > NMSThreash)
                detections[j].confidence = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceBox& det)
    { return det.confidence == 0; }), detections.end());
}

ImageDetector::ImageDetector(const std::string &onnxFile) : TrtPipeline(onnxFile) {
    Dims mInputDims = mEngine->getTensorShape(INPUT_NAME.c_str());
    mInputH = mInputDims.d[2];
    mInputW = mInputDims.d[3];
}

ImageDetector::~ImageDetector() {

}

void ImageDetector::_preprocessInput(std::shared_ptr<BufferManager> buffers, cv::Mat &img) {
    float ratio = float(mInputW) / float(img.cols) < float(mInputH) / float(img.rows) ? 
        float(mInputW) / float(img.cols) : float(mInputH) / float(img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(mInputW, mInputH), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    //HWC TO CHW
    int channelLength = mInputW * mInputH;
    float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer(INPUT_NAME));
    std::vector<cv::Mat> split_img = {
            cv::Mat(mInputH, mInputW, CV_32FC1, hostDataBuffer + channelLength * 2),
            cv::Mat(mInputH, mInputW, CV_32FC1, hostDataBuffer + channelLength * 1),
            cv::Mat(mInputH, mInputW, CV_32FC1, hostDataBuffer)
            };
    cv::split(flt_img, split_img);
}

void ImageDetector::_postprocessOutput(std::shared_ptr<BufferManager> buffers, cv::Mat &img) {
    float* heatmapBuffer = static_cast<float*>(buffers->getHostBuffer(OUTPUT_HEATMAP));
    float* scaleBuffer = static_cast<float*>(buffers->getHostBuffer(OUTPUT_SCALE));
    float* offsetBuffer = static_cast<float*>(buffers->getHostBuffer(OUTPUT_OFFSET));
    float* landmarksBuffer = static_cast<float*>(buffers->getHostBuffer(OUTPUT_LANDMARKS));

    std::vector<FaceBox> result;
    int image_size = mInputW / 4 * mInputH / 4;
    float ratio = float(img.cols) / float(mInputW) > float(img.rows) / float(mInputH) ? float(img.cols) / float(mInputW) : float(img.rows) / float(mInputH);
    for (int i = 0; i < mInputH / 4; i++) {
        for (int j = 0; j < mInputW / 4; j++) {
            int current = i * mInputW / 4 + j;
            if (heatmapBuffer[current] > confThreash) {
                FaceBox headbox;
                headbox.confidence = heatmapBuffer[current];
                headbox.h = std::exp(scaleBuffer[current]) * 4 * ratio;
                headbox.w = std::exp((scaleBuffer+image_size)[current]) * 4 * ratio;
                headbox.x = ((float)j + offsetBuffer[current] + 0.5f) * 4 * ratio;
                headbox.y = ((float)i + (offsetBuffer+image_size)[current] + 0.5f) * 4 * ratio;
                result.push_back(headbox);
            }
        }
    }
    NmsDetect(result);
    for (const auto& boundingBox : result) {
        cv::Rect box(boundingBox.x - boundingBox.w / 2, boundingBox.y - boundingBox.h / 2, boundingBox.w, boundingBox.h);
        cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2);
    }
}
