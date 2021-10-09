//
// Created by lxn on 21-7-28.
//

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "DeepLabV3P.h"

int main(int argc, char** argv)
{

    std::string wtsPath = "../res/deeplabv3.wts";
    std::string enginePath = "../res/deeplabv3.engine";
    std::string filepath = "../samples/";

    DeepLabV3P deepLabV3P;
    deepLabV3P.Init(wtsPath, enginePath);

    std::vector<std::string> files;
    TensorRTUtil::ListAllFiles(filepath, files);

    for(int k=0; k < files.size(); k++) {
        cv::Mat img = cv::imread(files[k]);
        cv::Mat mask_mat = deepLabV3P.Detect(img);
        cv::imwrite("output"+std::to_string(k)+".jpg", mask_mat);
    }

}
