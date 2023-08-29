#include "buffers.hpp"

using namespace nvinfer1;

BufferManager::BufferManager(std::shared_ptr<ICudaEngine> engine, 
    std::shared_ptr<IExecutionContext> context)
    : mEngine(engine) {
    for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
        std::string tensorName = std::string(mEngine->getIOTensorName(i));
        IOTensorNames.push_back(tensorName);

        // Create host and device buffers
        auto dims = context ? context->getTensorShape(tensorName.c_str()) : mEngine->getTensorShape(tensorName.c_str());
        auto dtype = engine->getTensorDataType(tensorName.c_str());
        int  size = 1;
        for (int j = 0; j < dims.nbDims; ++j)
            size *= dims.d[j];
        size = size * dataTypeToSize(dtype);
        std::unique_ptr<HostDeviceBuffer> hostDevBuf = std::make_unique<HostDeviceBuffer>();
        hostDevBuf->deviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>(size, dtype);
        hostDevBuf->hostBuffer = GenericBuffer<HostAllocator, HostFree>(size, dtype);
        mManagedBuffers[tensorName] = std::move(hostDevBuf);
    }
}

void* BufferManager::getBuffer(const bool isHost, const std::string& tensorName) {
    return isHost ? mManagedBuffers[tensorName]->hostBuffer.data() : mManagedBuffers[tensorName]->deviceBuffer.data();
}

void* BufferManager::getHostBuffer(const std::string& tensorName) {
    return getBuffer(true, tensorName);
}

 void* BufferManager::getDeviceBuffer(const std::string& tensorName) {
    return getBuffer(false, tensorName);
}

void BufferManager::copyInputToDevice() {
    for(const std::string &tensorName : IOTensorNames){
        if(mEngine->getTensorIOMode(tensorName.c_str()) == TensorIOMode::kINPUT){
            void* hostPtr = mManagedBuffers[tensorName]->hostBuffer.data();
            void* devPtr = mManagedBuffers[tensorName]->deviceBuffer.data();
            size_t size = mManagedBuffers[tensorName]->hostBuffer.nbBytes();
            cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
        }  
    }
} 

void BufferManager::copyOutputToHost() {
    for(const std::string &tensorName : IOTensorNames){
        if(mEngine->getTensorIOMode(tensorName.c_str()) == TensorIOMode::kOUTPUT){
            void* hostPtr = mManagedBuffers[tensorName]->hostBuffer.data();
            void* devPtr = mManagedBuffers[tensorName]->deviceBuffer.data();
            size_t size = mManagedBuffers[tensorName]->hostBuffer.nbBytes();
            cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
        }  
    }
} 

void BufferManager::printInfo(std::shared_ptr<IExecutionContext> context) {
    for (const std::string &tensorName : IOTensorNames) {
        std::cout << std::string(mEngine->getTensorIOMode(tensorName.c_str()) == TensorIOMode::kINPUT ? "Input [" : "Output[");
        std::cout << tensorName << std::string("]-> ");
        std::cout << dataTypeToString(mEngine->getTensorDataType(tensorName.c_str())) << std::string(" ");
        if(!context)
            std::cout << shapeToString(mEngine->getTensorShape(tensorName.c_str())) << std::string(" ");
        else
            std::cout << shapeToString(context->getTensorShape(tensorName.c_str())) << std::string(" ");
        std::cout << std::endl;
    }
}

void BufferManager::configContextTensorAddress(std::shared_ptr<IExecutionContext> context) {
    for(const auto &item : mManagedBuffers)
        context->setTensorAddress(item.first.c_str(), item.second->deviceBuffer.data());
}


