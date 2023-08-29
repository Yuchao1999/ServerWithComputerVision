
#include "TrtPipeline.hpp"

using namespace nvinfer1;

TrtPipeline::TrtPipeline(const std::string onnxFile, bool isDynamic) {
    _isDynamic = isDynamic;
    _onnxModelFile = onnxFile;
    size_t sep_pos = _onnxModelFile.find_last_of(".");
    _trtModelFile = _onnxModelFile.substr(0, sep_pos) + ".plan";
    if(ifFileExists(_trtModelFile.c_str()))
        loadTrtModel();
    else
        loadOnnxModel();
    if (mEngine == nullptr) 
        spdlog::error("[TRT] : Failed loading engine!");
    spdlog::info("[TRT] : Succeeded loading engine!");
}

TrtPipeline::~TrtPipeline() {

}

void TrtPipeline::loadTrtModel() {
    spdlog::info("Load TensorRT model : {}", _trtModelFile);

    // create engine from TensorRT .plan file
    std::ifstream engineFile(_trtModelFile, std::ios::binary);
    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineBinaryData(fsize);
    engineFile.read(engineBinaryData.data(), fsize);
    engineFile.close();

    mRuntime = std::shared_ptr<IRuntime>(createInferRuntime(gLogger)); 
    mEngine = std::shared_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(engineBinaryData.data(), fsize));
}

void TrtPipeline::loadOnnxModel() {
    spdlog::info("[TRT] : Build TensorRT model from {}.", _onnxModelFile);

    // create engine from onnx file
    IBuilder *builder = createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig *config = builder->createBuilderConfig();
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(_onnxModelFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    if(_isDynamic){
    // only support one input dynamic
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        ITensor *inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims{4, {1, 256, 256, 3}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims{4, {1, 512, 512, 3}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims{4, {1, 1024, 1024, 3}});
        config->addOptimizationProfile(profile);
    }
    IHostMemory *engineBinaryData = builder->buildSerializedNetwork(*network, *config);

    mRuntime = std::shared_ptr<IRuntime>(createInferRuntime(gLogger)); 
    mEngine = std::shared_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(engineBinaryData->data(), engineBinaryData->size()));

    // save the serialize engine to disk
    std::ofstream engineFile(_trtModelFile, std::ios::binary);
    engineFile.write(static_cast<char*>(engineBinaryData->data()), engineBinaryData->size());
    if(engineFile.fail())
        spdlog::error("[TRT] : Failed saving .plan file!");
    spdlog::info("[TRT] : Succeeded saving .plan file!");
    engineFile.close();
}

void TrtPipeline::inference(cv::Mat &image, std::shared_ptr<BufferManager> buffers, std::shared_ptr<IExecutionContext> context) {
    // Read the input data into the managed buffers
    _preprocessInput(buffers, image);

    // Memcpy from host input buffers to device input buffers
    buffers->copyInputToDevice();

    // Start executing the inference
    context->enqueueV3(0);

    // Memcpy from device output buffers to host output buffers
    buffers->copyOutputToHost();

    // Decode the output
    _postprocessOutput(buffers, image);
}

std::shared_ptr<BufferManager> TrtPipeline::createBuffer() {
    return std::make_shared<BufferManager>(mEngine);
}

std::shared_ptr<BufferManager> TrtPipeline::createBuffer(std::shared_ptr<IExecutionContext> context) {
    return std::make_shared<BufferManager>(mEngine, context);
}

std::shared_ptr<IExecutionContext> TrtPipeline::createContext() {
    return static_cast<std::shared_ptr<IExecutionContext>>(mEngine->createExecutionContext());
}

std::shared_ptr<IExecutionContext> TrtPipeline::createContext(cv::Size size) {
    std::shared_ptr<IExecutionContext> context = static_cast<std::shared_ptr<IExecutionContext>>(mEngine->createExecutionContext());
    context->setInputShape(mEngine->getIOTensorName(0), Dims{4, {1, size.height, size.width, 3}});
    return context;
}

