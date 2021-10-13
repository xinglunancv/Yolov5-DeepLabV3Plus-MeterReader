//
// Created by lxn on 21-8-26.
//

#ifndef CPP_TENSORRTUTIL_H
#define CPP_TENSORRTUTIL_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include "iostream"
#include <map>
#include <chrono>
#include "cmath"
#include <dirent.h>

namespace TensorRTUtil{

    void ListAllFiles(std::string path, std::vector<std::string> &files, std::vector<std::string> &filenames);

    // 判断文件是否存在
    bool IsFileExists(const std::string &filePath);

    // 等比例缩放图像
    cv::Mat ResizeRatio(cv::Mat& img, int input_h, int input_w);

    /////////////////////////////////// 加载权重 ///////////////////////////////////////////////

    std::map<std::string, nvinfer1::Weights> LoadWeights(const std::string file);


    //////////////////////////////////  通用的网络结构模块 /////////////////////////////////////

    nvinfer1::IScaleLayer* AddBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps);

    nvinfer1::IActivationLayer* CBR(nvinfer1::INetworkDefinition *network,
                                    std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                    int outch, int kernel_size, int stride, int padding, int dilation, std::string convName, std::string bnName);

    nvinfer1::IDeconvolutionLayer* Upsample(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor& input, int outch, int input_h, int input_w);

    /////////////////////////////////// 语义分割 后处理 //////////////////////////////////////
    void PostProcessing(float *mask, float *prob, int input_h, int input_w, int class_num);
}

#endif //CPP_TENSORRTUTIL_H
