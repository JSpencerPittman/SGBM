#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

void printMinMax(cv::Mat& mat) {
    double min, max;
    cv::minMaxIdx(mat, &min, &max);
    printf("Min: %lf, Max: %lf\n", min, max);
}

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4)
        throw std::runtime_error("Missing arguments: path to goal and predicted disparity map paths, respectively.");

    std::filesystem::path goalDisparityMapPath(argv[1]);
    std::filesystem::path predDisparityMapPath(argv[2]);

    // Load images
    cv::Mat goalDisparityMap = cv::imread(goalDisparityMapPath, cv::IMREAD_GRAYSCALE);
    cv::Mat predDisparityMap = cv::imread(predDisparityMapPath, cv::IMREAD_GRAYSCALE);

    int cropWidth = goalDisparityMap.cols - predDisparityMap.cols;
    int numElements = predDisparityMap.cols * predDisparityMap.rows;

    double l2Error = 0.0;
    for(int r = 0; r < predDisparityMap.rows; ++r) {
        for(int c = 0; c < predDisparityMap.cols; ++c) {
            uchar pred = predDisparityMap.at<uchar>(r, c);
            uchar goal = goalDisparityMap.at<uchar>(r, cropWidth + c);
            l2Error += std::pow(static_cast<double>(pred) - static_cast<double>(goal), 2);
        }
    }
    l2Error = std::pow(l2Error / static_cast<double>(numElements), 0.5);

    printf("L2 Error: %lf\n", l2Error);

    double within3Px = 0;
    for(int r = 0; r < predDisparityMap.rows; ++r) {
        for(int c = 0; c < predDisparityMap.cols; ++c) {
            uchar pred = predDisparityMap.at<uchar>(r, c);
            uchar goal = goalDisparityMap.at<uchar>(r, cropWidth + c);
            within3Px += (static_cast<double>(pred) - static_cast<double>(goal)) <= 3 ? 1 : 0;
        }
    }
    within3Px /= static_cast<double>(numElements);

    printf("<3px: %lf\n", within3Px);

    return 0;
}