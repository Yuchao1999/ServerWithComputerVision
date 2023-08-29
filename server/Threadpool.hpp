#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <iostream>
#include <vector>
#include <pthread.h>
#include <functional>
#include <queue>

#include "Datachannel.hpp"

const int MAX_THREADS = 64;
const int MAX_QUEUE = 1024;
const int DEFAULT_THREADS = 10;

typedef enum {
    running = 0,
    stopped = 1
} PoolState;

struct ThreadPoolTask {
    std::function<void(void *)> func;
    void *arg;
};

class ThreadPool {
private:
    std::vector<pthread_t> _threadPool;
    std::queue<ThreadPoolTask> _taskQue;
    pthread_mutex_t _mtx;
    pthread_cond_t _condv;
    PoolState _shutdown; 
    int _poolSize;
public:
    ThreadPool(int poolSize);
    ~ThreadPool();
    void threadPoolAdd(std::function<void(void *)> func, void *arg);
    void threadPoolDestroy();

    static void* start_thread(void* args);
    void worker();
};

#endif