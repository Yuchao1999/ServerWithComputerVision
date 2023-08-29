#ifndef SERVER_HPP
#define SERVER_HPP

#include <iostream>
#include <string.h>
#include <queue>
#include <map>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <fcntl.h>

#include <spdlog/spdlog.h>

#include "Epoll.hpp"
#include "Datachannel.hpp"
#include "Threadpool.hpp"
#include "utils.hpp"

class DataChannel;
class ThreadPool;
class ImageServer;

typedef enum  {
    IMAGE_DETECTION,
    IMAGE_GENERATION,
} TaskMode;

struct TaskConfig {
    TaskMode taskMode;
    ImageServer *server;
    cv::Size imgSize;
};

/*ImageServer class create a TCP server to accept connections.
  ImageServer need to */

class ImageServer
{
private:
    int _port;
    int _listenFd;
    struct sockaddr_in _servAddr;

    Epoll *_epoller;
    ThreadPool *_threadPool;
    std::map<int, DataChannel *> _connections;

    ThreadSafeQueue<ImageDetector *> _detectorQue; // 没有初始化
    ThreadSafeQueue<ImageGenerator *> _generatorQue;  // 没有初始化
    
    ThreadSafeQueue<DataChannel *> _readQue;
public:
    ImageServer(int port);
    ~ImageServer();
    void run(); // start to accept connnections
    void handleNewConnection(); // handle new connections
    void deleteConnection(int fd);
    void handleReadEvent(int fd); // handle read event

    int setBlocking(int fd);
    int setnonBlocking(int fd);
    int setKeepAlive(int fd);

    DataChannel* getReadTask();
    ImageDetector* getDetector();
    void addDetector(ImageDetector *detector);
    ImageGenerator* getGenerator();
    void addGenerator(ImageGenerator *generator);
    void* getTrtModel(TaskMode taskMode);
    void addTrtModel(const TaskMode taskMode, void* trtModel);

    void addTaskToThreadPool(std::function<void(void *)> func, void *arg);
    void getServerInfo();
};

#endif