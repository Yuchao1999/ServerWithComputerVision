#include "Epoll.hpp"

Epoll::Epoll() {
    _epollFd = epoll_create1(0);
    if(_epollFd == -1)
        throw std::runtime_error("Epoll create failed.");
    _events.resize(MAX_EVENTS);
}

Epoll::~Epoll() {
    close(_epollFd);
    _epollFd = -1;
}

int Epoll::epollAdd(int fd, uint32_t evs) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.data.fd = fd;
    ev.events = evs;
    return epoll_ctl(_epollFd, EPOLL_CTL_ADD, fd, &ev);
}

int Epoll::epollMod(int fd, uint32_t evs) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.data.fd = fd;
    ev.events = evs;
    return epoll_ctl(_epollFd, EPOLL_CTL_MOD, fd, &ev);
}

int Epoll::epollDel(int fd, uint32_t evs) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    return epoll_ctl(_epollFd, EPOLL_CTL_DEL, fd, &ev);
}

int Epoll::wait(int timeout) {
    return epoll_wait(_epollFd, &_events[0], MAX_EVENTS, timeout);
}

int Epoll::getEventFd(size_t i) {
    if(i >= _events.size() || i < 0)
        throw std::runtime_error("Index i out of events array size.");
    return _events[i].data.fd;
}

uint32_t Epoll::getEvents(size_t i) {
    return _events[i].events;
}