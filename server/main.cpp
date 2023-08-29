#include <iostream>
#include <string>
#include <pthread.h>
#include <vector>
#include "Server.hpp"
#include "Threadpool.hpp"

void* handleRead(void* arg);
void* handleProcess(void* arg);
void* handleWrite(void* arg);

/*Global service logic:
    1. Initialize and start running the Image process server.
    2. The main thread on which the server runs only takes responsibility for 
        accepting new connections.
    3. Create several child threads to do data read task, data process task 
        and data write task.
    4. 
*/

int main(int argc, char *argv[]) {
    ImageServer *serv;

    // Open and initialize the Image server.
    try{
        serv = new ImageServer(5001);
        spdlog::info("Open server complete!");
        serv->getServerInfo();
    }
    catch(std::runtime_error &err) {
        spdlog::error("Open server failed! Error info: {}", err.what());
        exit(1);
    }
    
    // Create child threads to receive, send and process data.
    int thread_err = 0;
    pthread_t parseThread;
    try{
        thread_err = pthread_create(&parseThread, NULL, handleRead, serv);
        if(thread_err != 0)
            throw std::runtime_error("Parese thread create failed!");
    }
    catch(std::runtime_error &err) {
        spdlog::error(err.what());
        exit(1);
    }

    // Start to accept connections and handle events from the connections.
    spdlog::info("Start to accept connectios ...");
    serv->run();
    return 0;
}

void* handleRead(void* arg) {
    ImageServer *serv = (ImageServer*) arg;
    while(true){
        DataChannel* dataChannel = serv->getReadTask();
        TaskConfig conf;
        conf.taskMode = IMAGE_GENERATION;
        conf.imgSize = cv::Size(512, 512);
        conf.server = serv;
        std::function<void(void *)> func = std::bind(&DataChannel::handleImage, dataChannel, std::placeholders::_1);
        serv->addTaskToThreadPool(func, &conf);
    }
    return NULL;
}
