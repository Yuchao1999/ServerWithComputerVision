/* Reference to the TensorRT github samples. */
#ifndef TENSORRT_BUFFERS_HPP
#define TENSORRT_BUFFERS_HPP

#include <iostream>
#include <NvInfer.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include "utils.hpp"
#include <spdlog/spdlog.h>

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
    public:
        size_t mSize{0};
        nvinfer1::DataType mType;
        void* mBuffer;
        AllocFunc allocFn;
        FreeFunc freeFn;
    public:
        //! \brief Construct an empty buffer.
        GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            : mSize(0)
            , mType(type)
            , mBuffer(nullptr) {
        }

        //! \brief Construct a buffer with the specified allocation size in bytes.
        GenericBuffer(size_t size, nvinfer1::DataType type)
            : mSize(size)
            , mType(type) {
            if (!allocFn(&mBuffer, this->nbBytes()))
                throw std::bad_alloc();
        } 

        GenericBuffer& operator=(GenericBuffer&& buf) {
            if (this != &buf)
            {
                freeFn(mBuffer);
                mSize = buf.mSize;
                mType = buf.mType;
                mBuffer = buf.mBuffer;
                // Reset buf.
                buf.mSize = 0;
                buf.mBuffer = nullptr;
            }
            return *this;
        }

        //! \brief Returns pointer to underlying array.
        void* data() const {
            return mBuffer;
        }

        //! \brief Returns the size (in number of elements) of the buffer.
        size_t size() const {
            return mSize;
        }

        //! \brief Returns the size (in bytes) of the buffer.
        size_t nbBytes() const {
            return this->size() * dataTypeToSize(mType);
        }

        ~GenericBuffer() {
            freeFn(mBuffer);
        }
};

class DeviceAllocator {
    public:
        bool operator()(void** ptr, size_t size) const {
            return cudaMalloc(ptr, size) == cudaSuccess;
        }
};

class DeviceFree {
    public:
        void operator()(void* ptr) const {
            cudaFree(ptr);
        }
};

class HostAllocator {
    public:
        bool operator()(void** ptr, size_t size) const {
            *ptr = malloc(size);
            return *ptr != nullptr;
        }
};

class HostFree {
    public:
        void operator()(void* ptr) const {
            free(ptr);
        }
};

//! \brief  The HostDeviceBuffer class groups together a pair of corresponding device and host buffers.
class HostDeviceBuffer {
public:
    GenericBuffer<DeviceAllocator, DeviceFree> deviceBuffer;
    GenericBuffer<HostAllocator, HostFree> hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager {
    private:
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;   //!< The pointer to the engine
        std::vector<std::string> IOTensorNames;
        std::unordered_map<std::string, std::unique_ptr<HostDeviceBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers

        void* getBuffer(const bool isHost, const std::string& tensorName);

    public:
        //! \brief Create a BufferManager for handling buffer interactions with engine.
        BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
            std::shared_ptr<nvinfer1::IExecutionContext> context = std::shared_ptr<nvinfer1::IExecutionContext>(nullptr));

        //! \brief Returns the device buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        void* getDeviceBuffer(const std::string& tensorName);

        //! \brief Returns the host buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        void* getHostBuffer(const std::string& tensorName);

        //! \brief Copy the contents of input host buffers to input device buffers synchronously.
        void copyInputToDevice();

        //! \brief Copy the contents of output device buffers to output host buffers synchronously.
        void copyOutputToHost();

        //! \brief print IO tensors information
        void printInfo(std::shared_ptr<nvinfer1::IExecutionContext> context = std::shared_ptr<nvinfer1::IExecutionContext>(nullptr));

        //! \brief Set memory address for given input or output tensor corresponding to a context
        void configContextTensorAddress(std::shared_ptr<nvinfer1::IExecutionContext> context);
};

#endif
