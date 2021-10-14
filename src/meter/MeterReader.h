//
// Created by lxn on 2021/10/13.
//

#ifndef DEEPLAB_METERREADER_H
#define DEEPLAB_METERREADER_H

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

class MeterReader{
public:
    void drawMask(cv::Mat& img, cv::Mat& mask, float scale_value);
    float reader_process(cv::Mat& img, cv::Mat& mask);

private:
    std::string floatToString(const float &val);
    std::vector<int> argsort(const std::vector<int>& array);
    double getDistance(cv::Point2f point1, cv::Point2f point2);
    int thresholdByCategory(cv::Mat& src, cv::Mat& dst, int category) ;
    int thresholdByContour(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& contour);
    int getScaleLocation(cv::Mat& dial_mask, cv::Point2f *locations);
    int getCenterLocation(cv::Mat& dial_mask, cv::Point2f& center_location);
    int getMinAreaRectPoints(cv::Mat& pointer_mask, cv::Point2f *Ps);
    int getPointerVertexIndex(cv::Point2f& center_location, cv::Point2f *Ps, int& vertex_index);
    int getPointerLocation(cv::Mat& pointer_mask, cv::Point2f& center_location, cv::Point2f *pointer_location);
    float getAngleRatio(cv::Point2f *scale_locations, cv::Point2f &pointer_head_location, cv::Point2f &center_location);
    float getScaleValue(float angleRatio);
    void vis_for_test(cv::Mat& img, cv::Point2f *scale_locations, cv::Point2f *pointer_locations,
                      cv::Point2f &center_location, float scale_value);

};


#endif //DEEPLAB_METERREADER_H

