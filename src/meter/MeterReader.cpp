//
// Created by lixiangnan on 2021/10/13.
//

#include "MeterReader.h"

#define SCALE_BEGINNING -0.1
#define SCALE_END 0.9

std::string MeterReader::floatToString(const float &val)
{
    char* chCode;
    chCode = new char[20];
    sprintf(chCode, "%.2lf", val);
    std::string str(chCode);
    delete[]chCode;
    return str;
}

// 实现argsort功能
std::vector<int> MeterReader::argsort(const std::vector<int>& array){

    int size = array.size();
    std::vector<int> array_index(size, 0); // 初始化一个长度为size,值全为0的数组
    for (int i = 0; i < size; i++){
        array_index[i] = i;
    }
    // sort 第三个参数是Lambda表达式， 表达式[]内是传入的参数
    std::sort(array_index.begin(), array_index.end(), [&array] (int pos1, int pos2){
        return array[pos1] < array[pos2];
    });

    return array_index;
}

// 计算两点间的距离
double MeterReader::getDistance(cv::Point2f point1, cv::Point2f point2){

    double distance = sqrtf(powf(point1.x - point2.x, 2) + powf(point1.y - point2.y, 2));
    return distance;
}


// 二值化，输入是语义分割的src（单通道图像）、输出dst(和src相同形状）和指定类别ID
// 像素点的灰度值如果等于指定类别ID则像素值置为255，否则置为0
// cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
int MeterReader::thresholdByCategory(cv::Mat& src, cv::Mat& dst, int category) {
    uchar *p = NULL;
    uchar *q = NULL;
    // 基于指针的方式来读取CV::Mat
    for (int row = 0; row < src.rows; row++){
        p = src.ptr<uchar>(row);
        q = dst.ptr<uchar>(row);
        for (int col = 0; col < src.cols; col++){
            if(p[col] == category){
                q[col] = 255;
            }else{
                q[col] = 0;
            }
        }
    }
    return 0;
}

// 输入是一个二值化的图像，针对指定区域的像素置为0
int MeterReader::thresholdByContour(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& contour){
    uchar *p = NULL;
    uchar *q = NULL;
//    cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    for(size_t row = 0; row < src.rows; row++){
        p = src.ptr<uchar>(row);
        q = dst.ptr<uchar>(row);
        for(size_t col = 0; col < src.cols; col++){
            if (pointPolygonTest(contour,cv::Point(col,row),false) == 1){
                q[col] = 0;
            }else{
                q[col] = p[col];
            }
        }
    }
    return 0;
}




// 获取表盘的起点和终点位置
// 输入的是表盘二值化图像，注意表盘需要先做一次腐蚀操作，过滤掉零星的误检
int MeterReader::getScaleLocation(cv::Mat& dial_mask, cv::Point2f *locations){
    // 定义表盘的起点位置和终点位置
    cv::Point2f beginning(-1, -1);
    cv::Point2f end(-1, -1);

    uchar *tmp = NULL;
    for(int row = dial_mask.rows - 1; row >= 0; row--){
        tmp = dial_mask.ptr<uchar>(row);
        for(int col = 0; col < dial_mask.cols; col++){
            int value = tmp[col];
            if (value == 255){
                // 满足起点位置在图像左半侧，且起点位置未赋值
                if (col < dial_mask.cols / 2 && beginning.x == -1){
                    beginning.x = col;
                    beginning.y = row;
                }
                // 满足终点位置在图像右半侧，且终点位置未赋值
                if (col >= dial_mask.cols / 2 && end.x == -1){
                    end.x = col;
                    end.y = row;
                }
            }
        }
        if (beginning.x >= 0 && end.x >= 0){
            locations[0] = beginning;
            locations[1] = end;
            return 0;
        }
    }
    return -1;
}


// 获取仪表中心点位置
int MeterReader::getCenterLocation(cv::Mat& dial_mask, cv::Point2f& center_location){

    // 直径（找到仪表掩码最右侧像素点位置减去最左侧像素点位置的最大值，即直径）
    int diameter = 0;
    // 满足直径的区域通常会是一个矩形区域，所以需要得到上下左右四个索引
    std::vector<int> diameter_index(4, -1);
    uchar *tmp = NULL;
    for(int row = dial_mask.rows - 1; row >= 0; row--){
        tmp = dial_mask.ptr<uchar>(row);
        int left  = 0;
        int right = dial_mask.cols - 1;
        while(left < right){
            if(!tmp[left])      left++;
            if(!tmp[right])     right--;
            if(tmp[left] && tmp[right]) break;
        }

        if(diameter <= right - left){
            diameter_index[0] = row;
            if(diameter < right - left){
                diameter_index[1] = row;
            }
            diameter = right - left;
            diameter_index[2] = left;
            diameter_index[3] = right;
//            std::cout << diameter_index[0] << " " <<diameter_index[1] << " " <<diameter_index[2] << " " <<diameter_index[3] << " " <<std::endl;
        }
    }
    if (diameter > 0){
        center_location.x = (diameter_index[2] + diameter_index[3]) / 2;
        center_location.y = (diameter_index[0] + diameter_index[1]) / 2;
        return 0;
    }else{
        return -1;
    }
}

// 获取图像中最大轮廓的最小外接旋转框矩形的四个顶点位置。
// 之所以是最大轮廓，也是为了过滤掉零星的误检
int MeterReader::getMinAreaRectPoints(cv::Mat& pointer_mask, cv::Point2f *Ps){

    //一个轮廓是若干个连续的点的列表，这里定义的是轮廓的列表
    std::vector<std::vector<cv::Point>> contours;
    // 保存的是轮廓间关系
    std::vector<cv::Vec4i> hierarchy;
    // 这里模式使用的是仅找最外层轮廓
    cv::findContours(pointer_mask, contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());

    // 异常判定
    int len = contours.size();
    if (len <= 0) return -1;

    int index = -1;
    int max_min_rect_area = 0; // 最大的最小外接矩形面积
    for(int i = 0; i < len; i++){
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        if (rect.size.area() > max_min_rect_area){
            max_min_rect_area = rect.size.area();
            index = i;
        }
    }

    cv::RotatedRect rect = cv::minAreaRect(contours[index]);
    rect.points(Ps);
    return 0;
}

// 获取指针头部的点（指针是一个旋转矩形，头部有两个顶点，本函数获取的是该点顺时针方向为短边的点）
// 输入：指针的二值化图像，仪表盘中心，指针旋转矩形四个顶点
// 输出：指针旋转矩形四个顶点中指针头部顶点的索引
int MeterReader::getPointerVertexIndex(cv::Point2f& center_location, cv::Point2f *Ps, int& vertex_index){
    int max_distance = -1;
    for(int i = 0; i < 4; i++){
        // 顺时针方向为短边
        if(getDistance(Ps[i], Ps[(i + 1) % 4]) > getDistance(Ps[i], Ps[(i + 3) % 4])){
            // 指针头部
            if (max_distance <= getDistance(Ps[i], center_location)){
                max_distance = getDistance(Ps[i], center_location);
                vertex_index = i;
            }
        }
    }
    return 0;
}


int MeterReader::getPointerLocation(cv::Mat& pointer_mask, cv::Point2f& center_location, cv::Point2f *pointer_location){

    // 先根据指针二值化图像来找到最小外接旋转矩形顶点
    cv::Point2f Ps[4];
    int res = getMinAreaRectPoints(pointer_mask, Ps);
    if (res == -1) return res;

    // 根据矩形顶点找到头部的点的索引
    int vertex_index = -1;
    getPointerVertexIndex(center_location, Ps, vertex_index);

    // 将指针分为两部分，靠近仪表中心的部分二值化为0
    cv::Point2f left_midPointer((Ps[vertex_index].x + Ps[(vertex_index + 1) % 4].x) / 2.0, (Ps[vertex_index].y + Ps[(vertex_index + 1) % 4].y) / 2.0);
    cv::Point2f right_midPointer((Ps[(vertex_index + 2) % 4].x + Ps[(vertex_index + 3) % 4].x) / 2.0, (Ps[(vertex_index + 2) % 4].y + Ps[(vertex_index + 3) % 4].y) / 2.0);

    std::vector<cv::Point2f> contour = {
            left_midPointer,
            Ps[(vertex_index + 1) % 4],
            Ps[(vertex_index + 2) % 4],
            right_midPointer
    };

    thresholdByContour(pointer_mask, pointer_mask,  contour);

    // 再根据二值化后的图像重新找最小外接旋转矩形
    res = getMinAreaRectPoints(pointer_mask, Ps);
    if(res == -1) return res;
    getPointerVertexIndex(center_location, Ps, vertex_index);

    cv::Point2f head_midPointer((Ps[vertex_index].x + Ps[(vertex_index + 3) % 4].x) / 2.0, (Ps[vertex_index].y + Ps[(vertex_index + 3) % 4].y) / 2.0);
    cv::Point2f tail_midPointer((Ps[(vertex_index + 1) % 4].x + Ps[(vertex_index + 2) % 4].x) / 2.0, (Ps[(vertex_index + 1) % 4].y + Ps[(vertex_index + 2) % 4].y) / 2.0);
    pointer_location[0] = head_midPointer;
    pointer_location[1] = tail_midPointer;
    return 0;
}


// 角度法计算当前刻度占总刻度的比例
float MeterReader::getAngleRatio(cv::Point2f *scale_locations, cv::Point2f &pointer_head_location, cv::Point2f &center_location){

    // 刻度开始点与x轴正方向的夹角，刻度结束点与x轴正方向的夹角，刻度开始点与刻度结束点的夹角
    float beginning_x_angle = atan2(center_location.y - scale_locations[0].y,
                                    scale_locations[0].x - center_location.x);
    float end_x_angle = atan2(center_location.y - scale_locations[1].y,
                              scale_locations[1].x - center_location.x);
    float beginning_end_angle = 2 * CV_PI - (end_x_angle - beginning_x_angle);

    float pointer_x_angle = atan2(center_location.y - pointer_head_location.y,
                                     pointer_head_location.x - center_location.x);
    float beginning_pointer_angle = 0;
    if(pointer_head_location.y > center_location.y && pointer_head_location.x < center_location.x){
        beginning_pointer_angle = pointer_x_angle - beginning_x_angle;
    }else{
        beginning_pointer_angle = 2 * CV_PI - (pointer_x_angle - beginning_x_angle);
    }

    float angleRatio = beginning_pointer_angle / beginning_end_angle;
    return angleRatio;
}

float MeterReader::getScaleValue(float angleRatio){
    float range = SCALE_END - SCALE_BEGINNING;
    float value = range * angleRatio + SCALE_BEGINNING;
    return value;
}

void MeterReader::vis_for_test(cv::Mat& img, cv::Point2f *scale_locations, cv::Point2f *pointer_locations,
                  cv::Point2f &center_location, float scale_value){

    cv::Mat img_copy = img.clone();
    cv::Scalar red = cv::Scalar(0, 0, 255);
    cv::circle(img_copy, scale_locations[0], 2, red, 2);
    cv::circle(img_copy, scale_locations[1], 2, red, 2);
    cv::circle(img_copy, center_location, 5, red, 2);

    cv::line(img_copy, pointer_locations[0], pointer_locations[1], red, 3, 8);

    std::string scale_value_str = floatToString(scale_value);
    cv::putText(img_copy, scale_value_str, cv::Point(img.cols / 3, img.rows / 2),
                cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, 8);

    cv::imshow("vis for test", img_copy);
    cv::waitKey(5000);
}

void MeterReader::drawMask(cv::Mat& img, cv::Mat& mask, float scale_value){
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat dial_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    cv::Mat pointer_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    thresholdByCategory(mask, dial_mask, 1);
    thresholdByCategory(mask, pointer_mask, 2);

    cv::medianBlur(dial_mask, dial_mask,7);
    cv::medianBlur(pointer_mask, pointer_mask,7);

    cv::findContours(dial_mask, contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
    cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

    cv::findContours(pointer_mask, contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
    cv::drawContours(img, contours, -1, cv::Scalar(0, 255, 255), 2, 8);

    std::string scale_value_str = floatToString(scale_value);
    cv::putText(img, "value: " + scale_value_str, cv::Point(10, 50),
                cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 0, 0), 2, 8);
}

// 输入是mask, 一共有四个类别，其中0-3分别代表背景、表盘、指针和刻度
// 输出是刻度值
float MeterReader::reader_process(cv::Mat& img, cv::Mat& mask){

    cv::Mat dial_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    cv::Mat pointer_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    thresholdByCategory(mask, dial_mask, 1);
    thresholdByCategory(mask, pointer_mask, 2);

    // 对表盘要先做腐蚀操作
    cv::Mat kernel(5, 5, CV_8U, cv::Scalar(1));
    cv::erode(dial_mask, dial_mask, kernel);

    cv::Point2f scale_locations[2];
    getScaleLocation(dial_mask, scale_locations);

    cv::Point2f center_location;
    getCenterLocation(dial_mask, center_location);

    cv::Point2f pointer_locations[2];
    getPointerLocation(pointer_mask, center_location, pointer_locations);

    float angle_ratio = getAngleRatio(scale_locations, pointer_locations[0], center_location);
    float scale_value = getScaleValue(angle_ratio);

    // 可视化，自身测试使用
//    vis_for_test(img, scale_locations, pointer_locations, center_location, scale_value);

    return scale_value;
}



