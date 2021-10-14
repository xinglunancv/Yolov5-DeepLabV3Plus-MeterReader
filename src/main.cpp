//
// Created by lxn on 21-7-28.
//

#include <opencv2/opencv.hpp>
#include "DeepLabV3P.h"
#include "MeterReader.h"



int main(int argc, char** argv)
{

    std::string wtsPath = "../res/best_model.wts";
    std::string enginePath = "../res/best_model.engine";
    std::string filepath = "../samples/";

    DeepLabV3P deepLabV3P;
    deepLabV3P.Init(wtsPath, enginePath);

    std::vector<std::string> files;
    std::vector<std::string> filenames;
    TensorRTUtil::ListAllFiles(filepath, files, filenames);

    for(int k=0; k < files.size(); k++) {
        std::cout << files[k] << std::endl;
//        if (filenames[k] != "sf6_201.jpg") continue;

        cv::Mat img = cv::imread(files[k]);
        cv::Mat mask = deepLabV3P.Detect(img);

        MeterReader meterReader;
        float scale_value = meterReader.reader_process(img, mask);
        meterReader.drawMask(img, mask, scale_value);
        std::cout << scale_value << std::endl;
        cv::imwrite(filenames[k]+"_show.jpg", img);

    }

}

