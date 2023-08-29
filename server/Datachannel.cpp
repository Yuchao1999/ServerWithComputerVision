#include "Datachannel.hpp"

DataChannel::DataChannel(int sockfd) 
    : _sockfd(sockfd) {
    pthread_mutex_init(&_mtx, NULL);
}

DataChannel::~DataChannel() {
    spdlog::error("Free DataChannel of {}", _sockfd);
    pthread_mutex_destroy(&_mtx);
}

template <typename dataType>
bool recvAll(int sockfd, dataType *buf, size_t fileSize) {
    while (fileSize > 0)
    {
        ssize_t readbytes = recv(sockfd, buf, fileSize, 0);
        if(readbytes == -1){
            if(errno == EAGAIN || errno == EWOULDBLOCK)
                continue;
            else
                false;
        }
        fileSize -= readbytes;
        buf += readbytes;
    }
    return true;
}

template <typename dataType>
bool sendAll(int sockfd, const dataType *buf, size_t fileSize) {
    while (fileSize > 0)
    {
        ssize_t sendBytes= send(sockfd, buf, fileSize, 0);
        if(sendBytes == -1)
            return false;
        fileSize -= sendBytes;
        buf += sendBytes;
    }
    return true;
}

cv::Mat DataChannel::recvImage() {
    pthread_mutex_lock(&_mtx);
    // receive the size of image byte stream
    int image_size = 0;
    if(recv(_sockfd, &image_size, sizeof(int), 0) == -1 && errno == EAGAIN){
        pthread_mutex_unlock(&_mtx);
        return {};
    }

    spdlog::info("Recevied image size : {}", image_size);
    // receive the image byte stream
    uchar buf[image_size];
    memset(buf, 0, image_size);
    recvAll(_sockfd, buf, image_size);
    pthread_mutex_unlock(&_mtx);

    std::vector<uchar> encode_data(buf, buf + sizeof(buf) / sizeof(buf[0]));
    cv::Mat img = cv::imdecode(encode_data, cv::IMREAD_COLOR);
    if(img.empty())
        spdlog::error("Image decode error! Maybe receive image failed.");
    return img;
}

void DataChannel::sendImage(cv::Mat img) {
    std::vector<uchar> encode_data;
    cv::imencode(".png", img, encode_data);
    std::string str_encode(encode_data.begin(), encode_data.end());

    pthread_mutex_lock(&_mtx);
    int length = str_encode.size();
    if(send(_sockfd , &length, sizeof(int), 0) <= 0) 
        spdlog::error("Send image length failed.");
    if(sendAll(_sockfd, str_encode.c_str(), str_encode.size()))
        spdlog::info("Send image successful.");
    else 
        spdlog::info("Send image failed.");
    pthread_mutex_unlock(&_mtx);
}

void DataChannel::handleImage(void *args) {
    TaskConfig *conf = (TaskConfig *)args;
    spdlog::info("Start process image.");
    cv::Mat img = recvImage();
    cv::resize(img, img, conf->imgSize);
    if(img.empty())
        spdlog::warn("Receive image failed.");
    TrtPipeline *trtModel = (TrtPipeline *)(conf->server->getTrtModel(conf->taskMode));
    std::shared_ptr<nvinfer1::IExecutionContext> context = trtModel->createContext(conf->imgSize);
    std::shared_ptr<BufferManager> buffers = trtModel->createBuffer(context);
    buffers->configContextTensorAddress(context);
    trtModel->inference(img, buffers, context);
    conf->server->addTrtModel(conf->taskMode, trtModel);
    sendImage(img);
    spdlog::info("Image process finished.");
}

void DataChannel::handleVideo(void *args) {
    TaskConfig *conf = (TaskConfig *)args;
    TrtPipeline *trtModel = (TrtPipeline *)(conf->server->getTrtModel(conf->taskMode));
    std::shared_ptr<nvinfer1::IExecutionContext> context = trtModel->createContext(conf->imgSize);
    std::shared_ptr<BufferManager> buffers = trtModel->createBuffer(context);
    buffers->configContextTensorAddress(context);
    while(true){
        cv::Mat img = recvImage();
        if(img.empty()) continue;
        cv::resize(img, img, conf->imgSize);
        trtModel->inference(img, buffers, context);
        sendImage(img);
    }   
    conf->server->addTrtModel(conf->taskMode, trtModel);
    spdlog::info("Video process stopped.");
}
