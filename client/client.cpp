#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

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

int main() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd == -1)
        throw std::runtime_error("Create client socket error.");
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_port = htons(5001);
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if(connect(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1)
        throw std::runtime_error("Connect error.");
    std::cout << "Successfully connected to the server." << std::endl;

    cv::Mat image = cv::imread("./image/selfie.png", cv::IMREAD_COLOR);
    std::vector<uchar> encode_data;
    cv::imencode(".png", image, encode_data);
    std::string str_encode(encode_data.begin(), encode_data.end());

    int length = str_encode.size();
    if(send(sockfd , &length, sizeof(int), 0) <= 0)
        spdlog::error("Send image length failed.");

    if(sendAll(sockfd, str_encode.c_str(), str_encode.size()))
        spdlog::info("Send image successful.");
    else
        spdlog::info("Send image failed.");

    // receive the size of image byte stream
    int image_size = 0;
    recv(sockfd, &image_size, sizeof(int), 0);
    spdlog::info("Recevied image size : {}", image_size);

    // receive the image byte stream
    uchar buf[image_size];
    memset(buf, 0, image_size);
    recvAll(sockfd, buf, image_size);
    std::vector<uchar> encode(buf, buf + sizeof(buf) / sizeof(buf[0]));
    image = cv::imdecode(encode, cv::IMREAD_COLOR);
    if(image.empty())
        spdlog::error("Image decode error! Maybe receive image failed.");
    cv::imwrite("./image/recv.png", image);

    close(sockfd);
    return 0;
}

// int main() {
//     int sockfd = socket(AF_INET, SOCK_STREAM, 0);
//     if(sockfd == -1)
//         throw std::runtime_error("Create client socket error.");
//     struct sockaddr_in serverAddr;
//     memset(&serverAddr, 0, sizeof(serverAddr));
//     serverAddr.sin_port = htons(5001);
//     serverAddr.sin_family = AF_INET;
//     serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

//     if(connect(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1)
//         throw std::runtime_error("Connect error.");
//     std::cout << "Successfully connected to the server." << std::endl;

//     cv::VideoCapture cap("./image/multi_face.avi");
//     if (!cap.isOpened()) {
//         std::cout << "Something Wrong with the Camera!!" << std::endl;
//         return -1;
//     }
//     cv::namedWindow("output",0);

//     cv::Mat frame;
//     while (true) {
//         clock_t start_time = clock();
//         cap.read(frame);
//         // 图像数据转换为字符串并发送给客户端
//         std::vector<uchar> encode_data;
//         cv::imencode(".png", frame, encode_data);
//         std::string str_encode(encode_data.begin(), encode_data.end());

//         int length = str_encode.size();
//         send(sockfd , &length, sizeof(int), 0);
//         if(sendAll(sockfd, str_encode.c_str(), str_encode.size()))
//             spdlog::info("Send image successful.");
//         else 
//             spdlog::info("Send image failed.");

//         // receive the size of image byte stream
//         int image_size = 0;
//         recv(sockfd, &image_size, sizeof(int), 0);
//         spdlog::info("Recevied image size : {}", image_size);

//         // receive the image byte stream
//         uchar buf[image_size];
//         memset(buf, 0, image_size);
//         recvAll(sockfd, buf, image_size);
//         std::vector<uchar> encode(buf, buf + sizeof(buf) / sizeof(buf[0]));
//         frame = cv::imdecode(encode, cv::IMREAD_COLOR);
//         clock_t end_time = clock();
//         double duration = double(end_time - start_time) / CLOCKS_PER_SEC * 1000;
//         std::cout << "程序执行时间：" << duration << " 毫秒" << std::endl;
//         cv::imshow("output", frame); 
//         cv::waitKey(1);
//     }

//     close(sockfd); // guanbi de shi hou ye hui fa chu xinhao
//     return 0;
// }