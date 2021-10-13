//
// Created by lxn on 21-7-28.
//

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "DeepLabV3P.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>

//#include <vector>
//#include <iostream>
//#include <math.h>
//
using namespace cv;
using namespace std;

// 实现argsort功能
template<typename T> std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}


double getDistance (Point2f point1, Point2f point2)
{
    double distance = sqrtf(powf((point1.x - point2.x),2) + powf((point1.y - point2.y),2));
    return distance;
}

cv::Mat thresholdByClass(int class_id, cv::Mat mask){
    int INPUT_H = mask.rows;
    int INPUT_W = mask.cols;
    uchar *ptmp = NULL;
    cv::Mat mask_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
    for (int i = 0; i < INPUT_H; i++) {
        ptmp = mask_mat.ptr<uchar>(i);
        for (int j = 0; j < INPUT_W; j++) {
            int val = mask.at<uchar>(i,j);
            if (val == class_id) {
                ptmp[j] = 255;
            } else {
                ptmp[j] = 0;
            }
        }
    }
    return mask_mat;
}

void getStartAndEndLocation(cv::Mat& mask, cv::Point2f *res){
    Point2f left(-1, -1);
    Point2f right(-1, -1);
    // 从下向上
    for(int i = mask.rows - 1 ; i > 0; i--){
        // 从左至右
        for(int j = 0; j < mask.cols; j++){
            int val = mask.at<uchar>(i,j);
            if (val == 255){
                if (j < mask.cols / 2 && left.x == -1){
                    left.x = j;
                    left.y = i;
                }
                if (j > mask.cols / 2 && right.x == -1){
                    right.x = j;
                    right.y = i;
                }
            }
        }
        if (left.x > 0 && right.x > 0){
            res[0] = left;
            res[1] = right;
            break;
        }
    }
}


void getCenterLocation(cv::Mat& mask, cv::Point2f& res){

    // 从下向上
    int diameter = -1;
    int diameter_left = -1;
    int diameter_right = -1;
    int diameter_index = -1;
    int diameter_index_prev = -1;
    for(int i = mask.rows - 1 ; i > 0; i--){
        // 从左至右
        int left = 0;
        int right = mask.cols - 1;
        while(left < right){

            int left_val = mask.at<uchar>(i, left);
            int right_val = mask.at<uchar>(i, right);
            if (left_val == 0){
                left++;
            }
            if(right_val == 0){
                right--;
            }
            if(left_val != 0 && right_val != 0){
                break;
            }
        }

        if (diameter <= right - left){
            if(diameter == right - left){
                diameter_index = i;
            }else{
                diameter_index = i;
                diameter_index_prev = i;
            }
            diameter = right - left;
            diameter_left = left;
            diameter_right = right;
            std::cout << "diameter:" << diameter << ", diameter_index:" << diameter_index << ", diameter_right-left:" << diameter_right - diameter_left << std::endl;
        }
    }
    res.x = (diameter_right + diameter_left) / 2;
    res.y = (diameter_index + diameter_index_prev) / 2;

}

int getRotatedRectPoints(cv::Mat& mask, cv::Point2f *Ps){
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point());

    // 找到面积最大轮廓
    int max_area_index = -1;
    int max_area = 0;
    for(int i=0; i <contours.size();i++){
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        if(rect.size.area() > max_area){
            max_area = rect.size.area();
            max_area_index = i;
        }
    }

    if(max_area_index >= 0){
        cv::RotatedRect rect = cv::minAreaRect(contours[max_area_index]);
        rect.points(Ps);
    }else{
        return -1;
    }
    return 0;
}

int DelImageByContours(cv::Mat& pointer_mask, std::vector<Point2f >& del_con){
    for(int i = 0; i < pointer_mask.rows; i++){
        for(int j = 0; j < pointer_mask.cols; j++){
            if(pointPolygonTest(del_con,cv::Point(j,i),false) == 1){
                pointer_mask.at<uchar>(i, j) = 0;
            }
        }
    }
    return 0;
}

int getLongSidePoints(cv::Mat& mask, cv::Point2f& centerPointer, cv::Point2f *Ps, int& long_side_index){
    int max_distatnce = -1;
    for(int i = 0; i < 4; i++){
        if (getDistance(Ps[i], Ps[(i+1)%4]) > getDistance(Ps[i], Ps[(i+3)%4]) ){
            if(max_distatnce <= getDistance(Ps[i], centerPointer)){
                max_distatnce = getDistance(Ps[i], centerPointer);
                long_side_index = i;
            }
        }
    }
    return 0;
}

int getPointerLocation(cv::Mat& pointer_mask, cv::Point2f& centerPointer, cv::Point2f *head_tail){

    cv::Point2f Ps[4];
    int res = getRotatedRectPoints(pointer_mask, Ps);
    if (res == 0){
        int long_side_index = -1;
        getLongSidePoints(pointer_mask, centerPointer, Ps, long_side_index);

        Point2f center_1((Ps[long_side_index].x + Ps[(long_side_index+1)%4].x) / 2.0,  (Ps[long_side_index].y + Ps[(long_side_index+1)%4].y) / 2.0);
        Point2f center_2((Ps[(long_side_index+2)%4].x + Ps[(long_side_index+3)%4].x) / 2.0,  (Ps[(long_side_index+2)%4].y + Ps[(long_side_index+3)%4].y) / 2.0);

        std::vector<Point2f > del_con = {center_1, Ps[(long_side_index+1)%4], Ps[(long_side_index+2)%4], center_2};

        DelImageByContours(pointer_mask, del_con);

        int sub_res = getRotatedRectPoints(pointer_mask, Ps);
        if(sub_res == 0){
            long_side_index = -1;
            getLongSidePoints(pointer_mask, centerPointer, Ps, long_side_index);
            head_tail[0].x = (Ps[(long_side_index-1)%4].x + Ps[long_side_index].x) / 2;
            head_tail[0].y = (Ps[(long_side_index-1)%4].y + Ps[long_side_index].y) / 2;
            head_tail[1].x = (Ps[(long_side_index+1)%4].x + Ps[(long_side_index+2)%4].x) / 2;
            head_tail[1].y = (Ps[(long_side_index+1)%4].y + Ps[(long_side_index+2)%4].y) / 2;
        }else{
            return -1;
        }

    }else{
        return -1;
    }
    return 0;
}



float getAngleRatio(Point2f &dial_start, Point2f &dial_end, Point2f  &pointer_head, Point2f &center) {

    // 距离水平X轴的夹角
    float dial_start_angle = atan2(center.y - dial_start.y, dial_start.x - center.x); // [-PI, 0]
    float dial_end_angle = atan2(center.y - dial_end.y, dial_end.x - center.x);  // [-PI, 0]
    float included_angle = (2 * CV_PI - (dial_end_angle - dial_start_angle)) * 180.0 / CV_PI;
    std::cout << "-0--------------------" << included_angle << std::endl;

    float pointer_head_angle = atan2(center.y - pointer_head.y, pointer_head.x - center.x);
    float sub_included_angle = 0;
    if (pointer_head.y > center.y && pointer_head.x < center.x){
        sub_included_angle = (pointer_head_angle - dial_start_angle) * 180.0 / CV_PI;
    }else {
        sub_included_angle = (2 * CV_PI - (pointer_head_angle - dial_start_angle)) * 180.0 / CV_PI;
    }
    std::cout << "-1--------------------" << sub_included_angle << std::endl;
    float ratio = sub_included_angle / included_angle;
    std::cout << "-2--------------------" << ratio << std::endl;
    int ratio_int = ratio * 100;
    ratio = ratio_int / 100.0;
    std::cout << "-3--------------------" << ratio << std::endl;
    return ratio - 0.1;


}


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
        cv::Mat img = cv::imread(files[k]);
        cv::Mat mask = deepLabV3P.Detect(img);

        cv::Mat dial_mask = thresholdByClass(1, mask);
        cv::Mat pointer_mask = thresholdByClass(2, mask);

//        getPointerLocation(mask, img);
        cv::imwrite(filenames[k]+"_dial_mask.jpg", dial_mask);
//        cv::imwrite(filenames[k]+"_pointer_mask.jpg", pointer_mask);
        cv::imwrite(filenames[k]+"_ori.jpg", img);


        // 先做腐蚀操作，目的是过滤掉误检的像素
        // 这里有两个条件：1，是误检的像素大小是小于kernel*kernel的区域。2，正确的像素大小应大于kernel*kernel的区域
        cv::Mat kernel(5, 5, CV_8U, cv::Scalar(1));
        cv::erode(dial_mask, dial_mask, kernel);
//
        cv::Point2f start_end_pointer[2];
        getStartAndEndLocation(dial_mask, start_end_pointer);

        cv::Point2f center_pointer;
        getCenterLocation(dial_mask, center_pointer);

        cv::Point2f head_tail[2];
        getPointerLocation(pointer_mask, center_pointer, head_tail);
        cv::imwrite(filenames[k]+"_pointer_mask.jpg", pointer_mask);

        cv::circle(img, start_end_pointer[0], 2, cv::Scalar (0, 0, 255), 1);
        cv::circle(img, start_end_pointer[1], 2, cv::Scalar (0, 0, 255), 1);
        cv::circle(img, center_pointer, 5, cv::Scalar (0, 0, 255), 1);

        cv::line(img, head_tail[0], head_tail[1], cv::Scalar(0,0, 255), 3, 8);

        float ratio = getAngleRatio(start_end_pointer[0], start_end_pointer[1], head_tail[0], center_pointer);
        cv::putText(img, std::to_string(ratio), Point(300, 100), cv::FONT_HERSHEY_COMPLEX,
                    0.5, cv::Scalar(0, 0, 255), 2, 8);


        cv::imwrite(filenames[k]+"_start.jpg", img);
//        std::cout << diameter[0] << "," << diameter[1]  << std::endl;


//        cv::imshow("src", src);
//        cv::waitKey();

//        cv::imwrite(filenames[k]+"_mask.jpg", mask);
//


//        break;
//        cv::RNG rng(1234);
//        cv::Mat dst(src.size(),src.type());
//
//        vector<vector<Point>> contours;
//        vector<Vec4i> hierarchy;
//        findContours(src,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point());
//
//        for(int i=0; i <contours.size();i++){
//            cv::Scalar color = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
//            cv::drawContours(img,contours,i,color,4);
//
//            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
//            cv::Point2f Ps[4];
//            rect.points(Ps);
//            for(int i=0;i<4;i++) cv::line(img,Ps[i],Ps[(i+1)%4],cv::Scalar(0,0,255),4);
//        }
//
//
//
//
//        cv::imshow("img", img);
//        waitKey();

//        return 0;

//        cv::imwrite("output"+std::to_string(k)+".jpg", mask_mat);
    }

}


//
//float getAngelOfTwoVector(Point2f &pt1, Point2f &pt2, Point2f &c)
//{
//    float theta = atan2(pt1.y - c.y, pt1.x - c.x) - atan2(pt2.y - c.y, pt2.x - c.x);
//    if (theta > CV_PI)
//        theta -= 2 * CV_PI;
//    if (theta < -CV_PI)
//        theta += 2 * CV_PI;
//
//    theta = theta * 180.0 / CV_PI;
//    return theta;
//}
//
//int main() {
//    Point2f c(0, 0);
//    Point2f pt1(-35.35, -35.35);
//    Point2f pt2(0, 50);
//
//    float theta = getAngelOfTwoVector(pt1, pt2, c);
//
//    cout << "theta: " << theta << endl;
//    return 0;
//
//}