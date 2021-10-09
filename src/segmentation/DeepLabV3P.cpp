//
// Created by lxn on 21-8-26.
//

#include "DeepLabV3P.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static const int INPUT_H = 400;
static const int INPUT_W = 400;
static const int OUTPUT_NUM = 3;

nvinfer1::IActivationLayer* DeepLabV3P::aspp(nvinfer1::INetworkDefinition *network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input){

    nvinfer1::IActivationLayer* x1 = TensorRTUtil::CBR(network, weightMap, input, 256, 1, 1, 0, 1, "ASSP.aspp1.0.weight", "ASSP.aspp1.1");   // 256*14*14
    nvinfer1::IActivationLayer* x2 = TensorRTUtil::CBR(network, weightMap, input, 256, 3, 1, 6, 6, "ASSP.aspp2.0.weight", "ASSP.aspp2.1");   // 256*14*14
    nvinfer1::IActivationLayer* x3 = TensorRTUtil::CBR(network, weightMap, input, 256, 3, 1, 12, 12, "ASSP.aspp3.0.weight", "ASSP.aspp3.1"); // 256*14*14
    nvinfer1::IActivationLayer* x4 = TensorRTUtil::CBR(network, weightMap, input, 256, 3, 1, 18, 18, "ASSP.aspp4.0.weight", "ASSP.aspp4.1"); // 256*14*14

    std::cout <<"debug x4 dim==" << x4->getOutput(0)->getDimensions().d[0] << " "
              << x4->getOutput(0)->getDimensions().d[1] << " " << x4->getOutput(0)->getDimensions().d[2] << std::endl;

    // todo 这里限制了输入的大小
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(input, nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{input.getDimensions().d[1], input.getDimensions().d[2]});
    nvinfer1::IActivationLayer* x5_1 = TensorRTUtil::CBR(network, weightMap, *pool1->getOutput(0), 256, 1, 1, 0, 1, "ASSP.avg_pool.1.weight", "ASSP.avg_pool.2");
    nvinfer1::IDeconvolutionLayer* x5 = TensorRTUtil::Upsample(network, *x5_1->getOutput(0), 256, input.getDimensions().d[1], input.getDimensions().d[2]);

    std::cout <<"debug x5 dim==" << x5->getOutput(0)->getDimensions().d[0] << " "
              << x5->getOutput(0)->getDimensions().d[1] << " " << x5->getOutput(0)->getDimensions().d[2] << std::endl; // todo 确定0是 C , 1 是 H  2是 W

    // torch.cat([x1, x2, x3, x4, x5], dim=1)
    nvinfer1::ITensor* inputTensors[] = { x1->getOutput(0), x2->getOutput(0), x3->getOutput(0), x4->getOutput(0), x5->getOutput(0) };
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 5);

    nvinfer1::IActivationLayer* result = TensorRTUtil::CBR(network, weightMap, *cat->getOutput(0), 256, 1, 1, 0, 1, "ASSP.conv1.weight", "ASSP.bn1");
    return result;

}



nvinfer1::IActivationLayer* DeepLabV3P::bottleneck(nvinfer1::INetworkDefinition *network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int inch, int outch, int stride, std::string lname){

    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    // conv1 1*1 s=1 p=0 outputMaps=outch
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, nvinfer1::DimsHW{1, 1},
                                                                   weightMap[lname + "conv1.weight"], emptywts);
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1});
    // bn
    nvinfer1::IScaleLayer* bn1 = TensorRTUtil::AddBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
    // relu
    nvinfer1::IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);

    // conv1 3*3 s=s p=1 outputMaps=outch
    nvinfer1::IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, nvinfer1::DimsHW{3, 3},
                                                                   weightMap[lname + "conv2.weight"], emptywts);
    // todo 空洞卷积
    if(lname.find("layer4")!=std::string::npos){
        conv2->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv2->setPaddingNd(nvinfer1::DimsHW{2, 2});
        conv2->setDilationNd(nvinfer1::DimsHW{2, 2});
    }else{
        conv2->setStrideNd(nvinfer1::DimsHW{stride, stride});
        conv2->setPaddingNd(nvinfer1::DimsHW{1, 1});
    }
    // bn
    nvinfer1::IScaleLayer* bn2 = TensorRTUtil::AddBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);
    // relu
    nvinfer1::IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), nvinfer1::ActivationType::kRELU);

    // conv1 1*1 s=1 p=0 outputMaps=outch*4
    nvinfer1::IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch*4, nvinfer1::DimsHW{1, 1},
                                                                   weightMap[lname + "conv3.weight"], emptywts);
    conv3->setStrideNd(nvinfer1::DimsHW{1, 1});
    // bn
    nvinfer1::IScaleLayer* bn3 = TensorRTUtil::AddBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);


    // 与捷径层的Tensor相加, 需要判断捷径层有没有经过1×1卷积以及经过卷积有没有做下采样
    // inch == outch * 4  表示不需要经过1 × 1卷积 发生在每一个layer 除了第一个bottleneck的其他bottleneck
    nvinfer1::IElementWiseLayer* ew1;
    if(stride !=1 || inch != outch * 4){
        // conv1 1*1 s=s p=0 outputMaps=outch * 4
        nvinfer1::IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, nvinfer1::DimsHW{1, 1},
                                                                       weightMap[lname + "downsample.0.weight"], emptywts);
        if(lname.find("layer4")!=std::string::npos){
            conv4->setStrideNd(nvinfer1::DimsHW{1, 1});
        }else{
            conv4->setStrideNd(nvinfer1::DimsHW{stride, stride});
        }
        // bn
        nvinfer1::IScaleLayer* bn4 = TensorRTUtil::AddBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        // add todo 第三个参数支持两个Tensor的各类点对点操作
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    }
    nvinfer1::IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), nvinfer1::ActivationType::kRELU);
    return relu3;
}

nvinfer1::ICudaEngine* DeepLabV3P::createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                    nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string &wtsFile)
{
    // 创建network
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    // 创建input tensor. shape: {3, INPUT_H, INPUT_W}, name: INPUT_BLOB_NAME
    nvinfer1::ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims3(3, INPUT_H, INPUT_W));
    assert(data);

    // 加载wts文件, 得到weightMap
    std::map<std::string, nvinfer1::Weights> weightMap = TensorRTUtil::LoadWeights(wtsFile);

    /////////////////////////////// Backbone  resnet101 ///////////////////////////////////////////////////////////////

    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    // conv1 7*7 s=2 p=3 outputMaps=64
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, nvinfer1::DimsHW{7, 7},
                                                                   weightMap["backbone.layer0.0.weight"], emptywts);
    conv1->setStrideNd(nvinfer1::DimsHW{2, 2});
    conv1->setPaddingNd(nvinfer1::DimsHW{3, 3});

    std::cout <<"debug  dim==" << conv1->getOutput(0)->getDimensions().d[0] << " "
              << conv1->getOutput(0)->getDimensions().d[1] << " " << conv1->getOutput(0)->getDimensions().d[2] << std::endl;

    // bn
    nvinfer1::IScaleLayer* bn1 = TensorRTUtil::AddBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.layer0.1", 1e-5);
    // relu todo 通过第二个参数可设置不同的激活方式
    nvinfer1::IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    // maxpooling 3*3 s=2 p=1 todo 通过第二个参数可设置不同的池化方式
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX,
                                                           nvinfer1::DimsHW{3, 3});
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});
    pool1->setPaddingNd(nvinfer1::DimsHW{1, 1});

    std::cout <<"debug  dim==" << pool1->getOutput(0)->getDimensions().d[0] << " "
              << pool1->getOutput(0)->getDimensions().d[1] << " " << pool1->getOutput(0)->getDimensions().d[2] << std::endl;

    // layer 1 <3层>
    nvinfer1::IActivationLayer* relu2 = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.");
    nvinfer1::IActivationLayer* relu3 = bottleneck(network, weightMap, *relu2->getOutput(0), 256, 64, 1, "backbone.layer1.1.");
    nvinfer1::IActivationLayer* relu4 = bottleneck(network, weightMap, *relu3->getOutput(0), 256, 64, 1, "backbone.layer1.2."); // low level  256*56*56

    std::cout <<"debug  dim==" << relu4->getOutput(0)->getDimensions().d[0] << " "
              << relu4->getOutput(0)->getDimensions().d[1] << " " << relu4->getOutput(0)->getDimensions().d[2] << std::endl;

    // layer 2 <4层>
    nvinfer1::IActivationLayer* relu5 = bottleneck(network, weightMap, *relu4->getOutput(0), 256, 128, 2, "backbone.layer2.0.");
    nvinfer1::IActivationLayer* relu6 = bottleneck(network, weightMap, *relu5->getOutput(0), 512, 128, 1, "backbone.layer2.1.");
    nvinfer1::IActivationLayer* relu7 = bottleneck(network, weightMap, *relu6->getOutput(0), 512, 128, 1, "backbone.layer2.2.");
    nvinfer1::IActivationLayer* relu8 = bottleneck(network, weightMap, *relu7->getOutput(0), 512, 128, 1, "backbone.layer2.3."); // 512*28*28

    std::cout <<"debug  dim==" << relu8->getOutput(0)->getDimensions().d[0] << " "
              << relu8->getOutput(0)->getDimensions().d[1] << " " << relu8->getOutput(0)->getDimensions().d[2] << std::endl;

    // layer 3 <23层>
    nvinfer1::IActivationLayer* relu9 = bottleneck(network, weightMap, *relu8->getOutput(0), 512, 256, 2, "backbone.layer3.0.");
    nvinfer1::IActivationLayer* relu10 = bottleneck(network, weightMap, *relu9->getOutput(0), 1024, 256, 1, "backbone.layer3.1.");
    nvinfer1::IActivationLayer* relu11 = bottleneck(network, weightMap, *relu10->getOutput(0), 1024, 256, 1, "backbone.layer3.2.");
    nvinfer1::IActivationLayer* relu12 = bottleneck(network, weightMap, *relu11->getOutput(0), 1024, 256, 1, "backbone.layer3.3.");
    nvinfer1::IActivationLayer* relu13 = bottleneck(network, weightMap, *relu12->getOutput(0), 1024, 256, 1, "backbone.layer3.4.");
    nvinfer1::IActivationLayer* relu14  = bottleneck(network, weightMap, *relu13->getOutput(0), 1024, 256, 1, "backbone.layer3.5.");
    nvinfer1::IActivationLayer* relu15 = bottleneck(network, weightMap, *relu14->getOutput(0), 1024, 256, 1, "backbone.layer3.6.");
    nvinfer1::IActivationLayer* relu16 = bottleneck(network, weightMap, *relu15->getOutput(0), 1024, 256, 1, "backbone.layer3.7.");
    nvinfer1::IActivationLayer* relu17 = bottleneck(network, weightMap, *relu16->getOutput(0), 1024, 256, 1, "backbone.layer3.8.");
    nvinfer1::IActivationLayer* relu18 = bottleneck(network, weightMap, *relu17->getOutput(0), 1024, 256, 1, "backbone.layer3.9.");
    nvinfer1::IActivationLayer* relu19  = bottleneck(network, weightMap, *relu18->getOutput(0), 1024, 256, 1, "backbone.layer3.10.");
    nvinfer1::IActivationLayer* relu20 = bottleneck(network, weightMap, *relu19->getOutput(0), 1024, 256, 1, "backbone.layer3.11.");
    nvinfer1::IActivationLayer* relu21 = bottleneck(network, weightMap, *relu20->getOutput(0), 1024, 256, 1, "backbone.layer3.12.");
    nvinfer1::IActivationLayer* relu22 = bottleneck(network, weightMap, *relu21->getOutput(0), 1024, 256, 1, "backbone.layer3.13.");
    nvinfer1::IActivationLayer* relu23 = bottleneck(network, weightMap, *relu22->getOutput(0), 1024, 256, 1, "backbone.layer3.14.");
    nvinfer1::IActivationLayer* relu24  = bottleneck(network, weightMap, *relu23->getOutput(0), 1024, 256, 1, "backbone.layer3.15.");
    nvinfer1::IActivationLayer* relu25 = bottleneck(network, weightMap, *relu24->getOutput(0), 1024, 256, 1, "backbone.layer3.16.");
    nvinfer1::IActivationLayer* relu26 = bottleneck(network, weightMap, *relu25->getOutput(0), 1024, 256, 1, "backbone.layer3.17.");
    nvinfer1::IActivationLayer* relu27 = bottleneck(network, weightMap, *relu26->getOutput(0), 1024, 256, 1, "backbone.layer3.18.");
    nvinfer1::IActivationLayer* relu28 = bottleneck(network, weightMap, *relu27->getOutput(0), 1024, 256, 1, "backbone.layer3.19.");
    nvinfer1::IActivationLayer* relu29  = bottleneck(network, weightMap, *relu28->getOutput(0), 1024, 256, 1, "backbone.layer3.20.");
    nvinfer1::IActivationLayer* relu30 = bottleneck(network, weightMap, *relu29->getOutput(0), 1024, 256, 1, "backbone.layer3.21.");
    nvinfer1::IActivationLayer* relu31 = bottleneck(network, weightMap, *relu30->getOutput(0), 1024, 256, 1, "backbone.layer3.22."); // 1024*14*14

    std::cout <<"debug  dim==" << relu31->getOutput(0)->getDimensions().d[0] << " "
              << relu31->getOutput(0)->getDimensions().d[1] << " " << relu31->getOutput(0)->getDimensions().d[2] << std::endl;

    // layer 4 <3层>
    nvinfer1::IActivationLayer* relu32 = bottleneck(network, weightMap, *relu31->getOutput(0), 1024, 512, 2, "backbone.layer4.0.");
    nvinfer1::IActivationLayer* relu33 = bottleneck(network, weightMap, *relu32->getOutput(0), 2048, 512, 1, "backbone.layer4.1.");
    nvinfer1::IActivationLayer* relu34 = bottleneck(network, weightMap, *relu33->getOutput(0), 2048, 512, 1, "backbone.layer4.2.");  // high level 2048*14*14

    std::cout <<"debug  dim==" << relu34->getOutput(0)->getDimensions().d[0] << " "
              << relu34->getOutput(0)->getDimensions().d[1] << " " << relu34->getOutput(0)->getDimensions().d[2] << std::endl;

    /////////////////////////////// ASPP //////////////////////////////////////////////////////////////////////////////

    nvinfer1::IActivationLayer* aspp_res = aspp(network, weightMap, *relu34->getOutput(0)); // 256*14*14

    std::cout <<"debug aspp_res dim==" << aspp_res->getOutput(0)->getDimensions().d[0] << " "
              << aspp_res->getOutput(0)->getDimensions().d[1] << " " << aspp_res->getOutput(0)->getDimensions().d[2] << std::endl;

    /////////////////////////////// Decoder //////////////////////////////////////////////////////////////////////////////

    nvinfer1::IActivationLayer* low_level_features = TensorRTUtil::CBR(network, weightMap, *relu4->getOutput(0), 48, 1, 1, 0, 1, "decoder.conv1.weight", "decoder.bn1");// 48*56*56

    std::cout <<"debug low_level_features dim==" << low_level_features->getOutput(0)->getDimensions().d[0] << " "
              << low_level_features->getOutput(0)->getDimensions().d[1] << " " << low_level_features->getOutput(0)->getDimensions().d[2] << std::endl;


    nvinfer1::IDeconvolutionLayer* aspp_res_upsample = TensorRTUtil::Upsample(network, *aspp_res->getOutput(0), 256, 4, 4); // 256*56*56

    nvinfer1::ITensor* inputTensors[] = { low_level_features->getOutput(0), aspp_res_upsample->getOutput(0) };
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 2); // (256+48)*56*56

    std::cout <<"debug cat dim==" << cat->getOutput(0)->getDimensions().d[0] << " "
              << cat->getOutput(0)->getDimensions().d[1] << " " << cat->getOutput(0)->getDimensions().d[2] << std::endl;

    nvinfer1::IActivationLayer* decoder_relu1 = TensorRTUtil::CBR(network, weightMap, *cat->getOutput(0), 256, 3, 1, 1, 1, "decoder.output.0.weight", "decoder.output.1"); // 256*56*56
    nvinfer1::IActivationLayer* decoder_relu2 = TensorRTUtil::CBR(network, weightMap, *decoder_relu1->getOutput(0), 256, 3, 1, 1, 1, "decoder.output.3.weight", "decoder.output.4");// 256*56*56

    std::cout <<"debug decoder_relu2 dim==" << decoder_relu2->getOutput(0)->getDimensions().d[0] << " "
              << decoder_relu2->getOutput(0)->getDimensions().d[1] << " " << decoder_relu2->getOutput(0)->getDimensions().d[2] << std::endl;

    nvinfer1::IConvolutionLayer* decoder_conv = network->addConvolutionNd(*decoder_relu2->getOutput(0), OUTPUT_NUM, nvinfer1::DimsHW{1, 1},
                                                                          weightMap["decoder.output.7.weight"], weightMap["decoder.output.7.bias"]);// OUTPUT_SIZE*56*56
    decoder_conv->setStrideNd(nvinfer1::DimsHW{1, 1});



    std::cout <<"debug decoder_conv dim==" << decoder_conv->getOutput(0)->getDimensions().d[0] << " "
              << decoder_conv->getOutput(0)->getDimensions().d[1] << " " << decoder_conv->getOutput(0)->getDimensions().d[2] << std::endl;


    nvinfer1::IDeconvolutionLayer* decoder_upsample = TensorRTUtil::Upsample(network, *decoder_conv->getOutput(0), OUTPUT_NUM, 4, 4); // OUTPUT_SIZE*224*224

    nvinfer1::ISoftMaxLayer* softmax1 = network->addSoftMax(*decoder_upsample->getOutput(0));
    nvinfer1::Dims dim0 = softmax1->getOutput(0)->getDimensions();
    std::cout <<"debug  softmax1 dim==" << dim0.d[0] << " " << dim0.d[1] << " " << dim0.d[2] << " " << dim0.d[3] << std::endl;

    softmax1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out " << std::endl;
    network->markOutput(*softmax1->getOutput(0));

    // 构建engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // todo 16M

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // destroy network
    network->destroy();

    // release host memory
    for(auto& mem: weightMap){
        free((void*) (mem.second.values));
    }

    return engine;
}


void DeepLabV3P::APIToModel(unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream, std::string &wtsFile){
    // 创建builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    // 创建config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // 基于API的方式搭建前向推理网络, 并生成引擎
    nvinfer1::ICudaEngine* engine = createEngine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wtsFile);
    assert(engine != nullptr);

    // 序列化
    (*modelStream) = engine->serialize();

    // destory();
    engine->destroy();
    builder->destroy();
    config->destroy();
}


// 语义分割的前向推理过程
void DeepLabV3P::doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchsize){

    const nvinfer1::ICudaEngine& engine = context.getEngine();

    // 指向要传递给引擎的输入和输出设备缓冲区的指针。
    // 引擎需要准确的 IEngine::getNbBindings() 数量的缓冲区。
    // todo 绑定索引的操作应该分别是 input_blob_name 和 output_blob_name
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // 为了绑定缓冲区，我们需要知道输入和输出张量的名称。
    // 请注意，索引保证小于 IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // 创建GPU buffers
    CHECK(cudaMalloc(&buffers[inputIndex], batchsize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchsize * OUTPUT_NUM * INPUT_H * INPUT_W * sizeof(float)));

    // 创建stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA 向设备输入批处理数据，异步推断批处理，并将 DMA 输出回主机
    // 通过cudaMemcpyAsync()在GPU与主机之间复制数据
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchsize * 3 * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchsize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchsize * OUTPUT_NUM * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    // 调用cudaStreamSynchronize()并指定想要等待的流：
    cudaStreamSynchronize(stream);

    // 释放资源
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int DeepLabV3P::GenerateEngine(std::string &wtsFile, std::string &enginePath){
    // 用于序列化引擎的一块内存区域
    nvinfer1::IHostMemory* modelStream{nullptr};
    // todo 这里的1是最大推理数
    APIToModel(1, &modelStream, wtsFile);
    assert(modelStream != nullptr);

    std::ofstream p(enginePath, std::ios::binary);
    if(!p){
        std::cerr << "不能打开计划输出的文件" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
}


int DeepLabV3P::Init(std::string &wtsFile, std::string &engineFile){

    if (!TensorRTUtil::IsFileExists(engineFile)){
        std::cout << "引擎不存在, 开始基于wtsFile来生成引擎!" << std::endl;
        if (!TensorRTUtil::IsFileExists(wtsFile)){
            std::cout << "wtsFile不存在!" << std::endl;
            return -1;
        }else{
            GenerateEngine(wtsFile,engineFile);
        }
    }


    char *trtModelStream{nullptr};
    size_t size{0};

    // 加载本地引擎文件, 读取到trtModelStream中
    std::ifstream file(engineFile, std::ios::binary);
    if(file.good()){
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    return 0;
}

cv::Mat DeepLabV3P::Detect(cv::Mat &img) {
    // 数据类型
    static float data[3 * INPUT_H * INPUT_W];
    static float prob[OUTPUT_NUM * INPUT_H * INPUT_W];
    float mask[INPUT_H * INPUT_W];

    cv::Mat pr_img = TensorRTUtil::ResizeRatio(img, INPUT_H, INPUT_W);

    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar *uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = ((float) uc_pixel[2] / 255.0 - 0.625) / 0.131;
            data[i + INPUT_H * INPUT_W] = ((float) uc_pixel[1] / 255.0 - 0.448) / 0.177;
            data[i + 2 * INPUT_H * INPUT_W] = ((float) uc_pixel[0] / 255.0 - 0.688) / 0.101;
            uc_pixel += 3;
            ++i;
        }
    }

    // 执行推理
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    TensorRTUtil::PostProcessing(mask, prob, INPUT_H, INPUT_W, OUTPUT_NUM);

    cv::Mat mask_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
    uchar *ptmp = NULL;
    for (int i = 0; i < INPUT_H; i++) {
        ptmp = mask_mat.ptr<uchar>(i);
        for (int j = 0; j < INPUT_W; j++) {
            float *pixcel = mask + i * INPUT_W + j;
            if (*pixcel == 0) {
                ptmp[j] = 0;
            } else if (*pixcel == 1) {
                ptmp[j] = 127;
            } else {
                ptmp[j] = 255;
            }
        }
    }

    return mask_mat;

}