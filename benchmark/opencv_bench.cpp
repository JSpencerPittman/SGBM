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
        throw std::runtime_error("Missing arguments: path to left and right image paths, respectively.");

    std::filesystem::path leftImagePath(argv[1]);
    std::filesystem::path rightImagePath(argv[2]);

    std::filesystem::path outputPath;
    if (argc == 4)
        outputPath = std::filesystem::path(argv[3]);
    else
        outputPath = "disparity.png";

    // Load images
    cv::Mat leftImage = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
    printf("Loaded left and right image.\n");

    // SGBM
    auto sgbm = cv::StereoSGBM::create(0, 65, 9, 7, 17, -1, -1, -1, 200, 2, cv::StereoSGBM::MODE_SGBM);
    cv::Mat disparityMap;
    sgbm->compute(leftImage, rightImage, disparityMap);

    // Rank-based Filtering
    cv::medianBlur(disparityMap, disparityMap, 5);
    cv::dilate(disparityMap, disparityMap, 7, {-1, -1}, 2);
    cv::erode(disparityMap, disparityMap, 7, {-1, -1}, 2);

    // Shift values
    for(int row = 0; row < disparityMap.rows; ++row)
        for(int col = 0; col < disparityMap.cols; ++col)
            disparityMap.at<short>(row, col) = disparityMap.at<short>(row, col) >> 4;

    disparityMap.convertTo(disparityMap, CV_8U);

    cv::imwrite(outputPath, disparityMap);
    printf("Saved Disparity Map To Disk.\n");

    return 0;
}