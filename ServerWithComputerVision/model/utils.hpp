#ifndef UTILS_TRT_HPP
#define UTILS_TRT_HPP

#include <string>
#include <NvInfer.h>
#include <sys/types.h>
#include <sys/stat.h>


inline bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}

inline std::string dataTypeToString(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return std::string("FP32 ");
    case nvinfer1::DataType::kHALF:
        return std::string("FP16 ");
    case nvinfer1::DataType::kINT8:
        return std::string("INT8 ");
    case nvinfer1::DataType::kINT32:
        return std::string("INT32");
    case nvinfer1::DataType::kBOOL:
        return std::string("BOOL ");
    default:
        return std::string("Unknown");
    }
}

inline std::string shapeToString(nvinfer1::Dims32 dim)
{
    std::string output("(");
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

inline size_t dataTypeToSize(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8: return 1;
    }
    return 0;
}

#endif