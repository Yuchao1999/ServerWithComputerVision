
#include "Threadpool.hpp"

ThreadPool::ThreadPool(int poolSize) {
    pthread_mutex_init(&_mtx, NULL);
    pthread_cond_init(&_condv, NULL);
    if(poolSize <= 0 || poolSize > MAX_THREADS)
        _poolSize = 8;
    _threadPool.resize(poolSize);
    _shutdown = running;
    for(int i = 0; i < poolSize; ++i)
        pthread_create(&_threadPool[i], NULL, start_thread, this);
}

ThreadPool::~ThreadPool() {
    // release resources
    threadPoolDestroy();
    pthread_mutex_destroy(&_mtx);
    pthread_cond_destroy(&_condv);
}

void ThreadPool::threadPoolAdd(std::function<void(void *)> func, void *arg) {
    if(_taskQue.size() > MAX_QUEUE){
        std::cout << "Task queue is full. Please wait a while and submit again." << std::endl;
        return;
    }
    ThreadPoolTask task;
    task.func = func;
    task.arg = arg;
    pthread_mutex_lock(&_mtx);
    _taskQue.push(task);
    pthread_cond_signal(&_condv);
    pthread_mutex_unlock(&_mtx);
}

void ThreadPool::threadPoolDestroy() {
    // Note: threadPoolDestroy will only be called by the main thread
    std::cout << "Destroy the thread pool." << std::endl;
    pthread_mutex_lock(&_mtx);
    _shutdown = stopped;
    pthread_mutex_unlock(&_mtx);
    std::cout << "Broadcasting shutdown signal to all threads..." << std::endl;
    pthread_cond_broadcast(&_condv);
    
    for(int i = 0; i < _threadPool.size(); ++i){
        pthread_join(_threadPool[i], NULL);
        pthread_cond_broadcast(&_condv);
    }
        
}

// We can't pass a member function to pthread_create.
// So created the wrapper function that calls the member function
// we want to run in the thread.
void* ThreadPool::start_thread(void* args) {
    ThreadPool* curPool = (ThreadPool*) args;
    curPool->worker();
    return NULL;
}

void ThreadPool::worker() {
    while(true) {
        pthread_mutex_lock(&_mtx);

        while(_taskQue.empty() && _shutdown != stopped)
            pthread_cond_wait(&_condv, &_mtx);

        if(_shutdown == stopped) {
            std::cout << "Tread exit : " << pthread_self() << std::endl;
            pthread_mutex_unlock(&_mtx);
            pthread_exit(NULL);
        }

        ThreadPoolTask task = _taskQue.front();
        _taskQue.pop();
        pthread_mutex_unlock(&_mtx);
        (task.func)(task.arg);  // execute the task
    }
}