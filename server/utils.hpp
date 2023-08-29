#ifndef UTILS_SERVER_HPP
#define UTILS_SERVER_HPP

template <typename dataType>
class ThreadSafeQueue {
private:
    pthread_mutex_t _mtx;
    pthread_cond_t _condv;
    std::queue<dataType> _queue;
public:
    ThreadSafeQueue() {
        pthread_mutex_init(&_mtx, NULL);
        pthread_cond_init(&_condv, NULL);
    }

    ~ThreadSafeQueue() {
        pthread_mutex_destroy(&_mtx);
        pthread_cond_destroy(&_condv);
    }

    void push(dataType data) {
        pthread_mutex_lock(&_mtx);
        _queue.push(data);
        pthread_cond_signal(&_condv);
        pthread_mutex_unlock(&_mtx);
    }

    dataType pop() {
        pthread_mutex_lock(&_mtx);
        if(_queue.empty())
            pthread_cond_wait(&_condv, &_mtx);
        dataType data = _queue.front();
        _queue.pop();
        pthread_mutex_unlock(&_mtx);
        return data;
    }

    bool isEmpty() {
        return _queue.empty();
    }
};

#endif