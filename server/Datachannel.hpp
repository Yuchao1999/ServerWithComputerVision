/*This header define a class for image receive, 
send and process operation.*/
#ifndef DATACHANNEL_HPP
#define DATACHANNEL_HPP

#include <iostream>
#include <string>
#include <sys/socket.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include "imageDetector.hpp"
#include "ImageGenerator.hpp"
#include "Server.hpp"
#include "utils.hpp"

class ImageServer;
class ThreadPool;
class DataChannel {
    private:
        int _sockfd;
        pthread_mutex_t _mtx;
    public:
        DataChannel(int sockfd);
        ~DataChannel();
        cv::Mat recvImage();
        void sendImage(cv::Mat img);
        void recvVideo(void *arg);
        int getSocketFd() { return _sockfd; }

    public:
        void handleImage(void *args);
        void handleVideo(void *args);

        void debug() {spdlog::error("Debug info.");}
};

#endif