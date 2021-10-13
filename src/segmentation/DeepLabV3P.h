//
// Created by lxn on 21-8-26.
//

#ifndef CPP_DEEPLABV3P_H
#define CPP_DEEPLABV3P_H

#include "TensorRTUtil.h"
#include "logging.h"

class DeepLabV3P{

public:

    int Init(std::string &wtsFile, std::string &engineFile);

    cv::Mat Detect(cv::Mat &img);


private:

    nvinfer1::IExecutionContext* context;

    nvinfer1::IActivationLayer* aspp(nvinfer1::INetworkDefinition *network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input);


    nvinfer1::IActivationLayer* bottleneck(nvinfer1::INetworkDefinition *network,
                                           std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                           int inch, int outch, int stride, std::string lname);

    nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                        nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string &wtsFile);


    void APIToModel(unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream, std::string &wtsFile);


    // 语义分割的前向推理过程
    void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchsize);

    int GenerateEngine(std::string &wtsFile, std::string &enginePath);

};



#endif //CPP_DEEPLABV3P_H
