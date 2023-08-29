#include "Server.hpp"

const int DEFAULT_DETECTOR_NUMS = 1;
const int DEFAULT_GENERATOR_NUMS = 1;

const std::string DETECTOR_ONNX = "./model/CenterFace/centerface_480_640.onnx";
const std::string GENERATOR_ONNX = "./model/AnimeGANv3/AnimeGANv3_PortraitSketch.onnx";

ImageServer::ImageServer(int port){
    if(port < 0 || port > 65535)
        throw std::runtime_error("Port out of bound.");
    _port = port;

    _listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if(_listenFd == -1)
        throw std::runtime_error("Server socket create failed.");

    memset(&_servAddr, 0, sizeof(_servAddr));
    _servAddr.sin_family = AF_INET;
    _servAddr.sin_port = htons(port);
    _servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    if(bind(_listenFd, (sockaddr *)&_servAddr, sizeof(_servAddr)) == -1)
        throw std::runtime_error("Bind failed.");

    if(listen(_listenFd, 128) == -1)
        throw std::runtime_error("Set listen available to listen failed.");
    
    _epoller = new Epoll();
    _epoller->epollAdd(_listenFd, EPOLLIN | EPOLLET);
    _threadPool = new ThreadPool(20);

    for(int i = 0; i < DEFAULT_DETECTOR_NUMS; ++i) {
        ImageDetector *detector = new ImageDetector(DETECTOR_ONNX);
        _detectorQue.push(detector);
    }

    for(int i = 0; i < DEFAULT_GENERATOR_NUMS; ++i) {
        ImageGenerator *generator = new ImageGenerator(GENERATOR_ONNX, true);
        _generatorQue.push(generator);
    }
}

ImageServer::~ImageServer() {
    delete _epoller;
    delete _threadPool;
    while(!_detectorQue.isEmpty()) {
        ImageDetector *detector = _detectorQue.pop();
        delete detector;
    }
    while(!_generatorQue.isEmpty()) {
        ImageGenerator *generator = _generatorQue.pop();
        delete generator;
    }
    spdlog::error("Image Server Shutdown.");
}

void ImageServer::getServerInfo() {
    spdlog::info("Server Socket File Discripter : {}", _listenFd);
    spdlog::info("Server IP : {}", inet_ntoa(_servAddr.sin_addr));
    spdlog::info("Server Port : {}", ntohs(_servAddr.sin_port));
}

void ImageServer::run() {
    while(true){
        int event_num = _epoller->wait(-1);
        for(int i = 0; i < event_num; ++i){
            int fd = _epoller->getEventFd(i);
            int event = _epoller->getEvents(i);
            if(fd == _listenFd) {
                try {
                    handleNewConnection();
                }
                catch(std::runtime_error &err) {
                    spdlog::error("Accept new connection error : {}", err.what());
                }
            }
            else if(event & EPOLLIN)
                handleReadEvent(fd);
        }
    }
}

int ImageServer::setnonBlocking(int fd) {
    int flags = fcntl(fd, F_GETFL);
    flags |= O_NONBLOCK;
    return fcntl(fd, F_SETFL, flags);
}

int ImageServer::setBlocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        return -1;
    }
    return fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
}

int ImageServer::setKeepAlive(int fd) {
    int flag = 0;
    int keepalive = 1;
    int keepidle = 60;
    int keepinterval = 5;
    int keepcount = 3;
    flag |= setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (void *)&keepalive , sizeof(keepalive ));
    flag |= setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, (void*)&keepidle , sizeof(keepidle ));
    flag |= setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, (void *)&keepinterval , sizeof(keepinterval ));
    flag |= setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, (void *)&keepcount , sizeof(keepcount ));
    return flag;
}

void ImageServer::handleNewConnection() {
    struct sockaddr_in clientAddr;
    memset(&clientAddr, 0, sizeof(clientAddr));
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientFd = -1;
    clientFd = accept(_listenFd, (sockaddr *)&clientAddr, &clientAddrLen);
    if(clientFd < 0)
        throw std::runtime_error("There is a connection failed to accept!");
    spdlog::info("New connection -> socket fd : {}, IP : {}, Port: {}", clientFd, inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));
    
    if(setnonBlocking(clientFd) == -1)
        throw std::runtime_error("Set new client socket non blocking failed!");

    if(setKeepAlive(clientFd) != 0)
        throw std::runtime_error("Set new client socket keepalive failed!");

    if(_epoller->epollAdd(clientFd, EPOLLIN | EPOLLET | EPOLLONESHOT) != 0)
        throw std::runtime_error("Add new client socket into Epoll failed.");

    DataChannel *dataChannel = new DataChannel(clientFd);
    _connections[clientFd] = dataChannel;
}

void ImageServer::deleteConnection(int fd) {
    DataChannel *dataChannel = _connections[fd];
    _connections.erase(fd);
    delete dataChannel;
}

void ImageServer::handleReadEvent(int fd) {
    spdlog::info("Handle a read event.");
    DataChannel *dataChannel = _connections[fd];
    _readQue.push(dataChannel);
}

DataChannel* ImageServer::getReadTask() {
    return _readQue.pop();
}

ImageDetector* ImageServer::getDetector() {
    return _detectorQue.pop();
}

void ImageServer::addDetector(ImageDetector *detector) {
    _detectorQue.push(detector);
}
    
ImageGenerator* ImageServer::getGenerator() {
    return _generatorQue.pop();
}

void ImageServer::addGenerator(ImageGenerator *generator) {
    _generatorQue.push(generator);
}

void* ImageServer::getTrtModel(TaskMode taskMode) {
    if(taskMode == IMAGE_DETECTION) {
        return getDetector();
    }
    else if(taskMode == IMAGE_GENERATION) {
        return getGenerator();
    }
    else {
        return nullptr;
    }
}

void ImageServer::addTrtModel(const TaskMode taskMode, void* trtModel) {
    if(taskMode == IMAGE_DETECTION) {
        ImageDetector *detector = (ImageDetector *)(trtModel);
        addDetector(detector);
    }
    else if(taskMode == IMAGE_GENERATION) {
        ImageGenerator *generator = (ImageGenerator *)(trtModel);
        addGenerator(generator);
    }
}

void ImageServer::addTaskToThreadPool(std::function<void(void *)> func, void *arg) {
    _threadPool->threadPoolAdd(func, arg);
}
