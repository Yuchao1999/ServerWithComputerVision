#ifndef EPOLL_HPP
#define EPOLL_HPP

#include <iostream>
#include <string.h>
#include <vector>
#include <sys/epoll.h>
#include <unistd.h>
#include <spdlog/spdlog.h>

class Epoll {
private:
    static const int MAX_EVENTS = 512;
    int _epollFd;
    std::vector<struct epoll_event> _events;
public:
    Epoll();
    ~Epoll();
    int getFd() { return _epollFd; }
    int epollAdd(int fd, uint32_t);
    int epollMod(int fd, uint32_t);
    int epollDel(int fd, uint32_t);
    int wait(int timeout);
    int getEventFd(size_t i);
    uint32_t getEvents(size_t i);
};

#endif