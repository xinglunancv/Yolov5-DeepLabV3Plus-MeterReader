//
// Created by lxn on 21-8-26.
//

#include "TensorRTUtil.h"



cv::Mat TensorRTUtil::ResizeRatio(cv::Mat& img, int input_h, int input_w) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {// img.cols > img.rows
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

bool TensorRTUtil::IsFileExists(const std::string &filePath) {
    if (FILE *file = fopen(filePath.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
};


// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> TensorRTUtil::LoadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout << "Loading weights successfully" << file << std::endl;
    return weightMap;
}


// BN层实现
nvinfer1::IScaleLayer* TensorRTUtil::AddBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;    // len 应该是通道数
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


// 一个完整的卷积块：Conv+BN+ReLU
nvinfer1::IActivationLayer* TensorRTUtil::CBR(nvinfer1::INetworkDefinition *network,
                                std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                int outch, int kernel_size, int stride, int padding, int dilation, std::string convName, std::string bnName){

    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    // conv1
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, nvinfer1::DimsHW{kernel_size, kernel_size},
                                                                   weightMap[convName], emptywts);
    conv1->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv1->setPaddingNd(nvinfer1::DimsHW{padding, padding});
    conv1->setDilationNd(nvinfer1::DimsHW{dilation, dilation});
    // bn
    nvinfer1::IScaleLayer* bn1 = AddBatchNorm2d(network, weightMap, *conv1->getOutput(0), bnName, 1e-5);
    // relu
    nvinfer1::IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    return relu1;
}


// binear upsample
nvinfer1::IDeconvolutionLayer* TensorRTUtil::Upsample(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor& input, int outch, int input_h, int input_w){

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * outch * input_h * input_w));
    for (int i = 0; i < outch * input_h * input_w; i++) {
        deval[i] = 1.0;
    }
    nvinfer1::Weights emptywts{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::Weights deconvwts{ nvinfer1::DataType::kFLOAT, deval, outch * input_h * input_w };
    nvinfer1::IDeconvolutionLayer* interpolate = network->addDeconvolutionNd(input, outch, nvinfer1::DimsHW{ input_h, input_w }, deconvwts, emptywts);
    interpolate->setStrideNd(nvinfer1::DimsHW{ input_h, input_w });
    interpolate->setNbGroups(outch);

    return interpolate;
}


// 语义分割输出的特征图的处理
void TensorRTUtil::PostProcessing(float *mask, float *prob, int input_h, int input_w, int class_num){
    int j = 0;
    for (int row = 0; row < input_h; ++row) {
        for (int col = 0; col < input_w; ++col) {
            int maxIndex = 0;
            for(int k = 0; k < class_num; k++){
                if(prob[j + maxIndex * input_h * input_w] < prob[j + k * input_h * input_w]){
                    maxIndex = k;
                }
            }
            mask[j] = maxIndex;
            ++j;
        }
    }
}

// 输出文件夹内所有文件
void TensorRTUtil::ListAllFiles(std::string path, std::vector<std::string> &files) {
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        return;
    }
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8)//file
            files.push_back(path + "/" + ptr->d_name);
        else if (ptr->d_type == 10)//link file
            continue;
        else if (ptr->d_type == 4) {//dir
            ListAllFiles(path + "/" + ptr->d_name, files);
        }
    }
    closedir(dir);
}