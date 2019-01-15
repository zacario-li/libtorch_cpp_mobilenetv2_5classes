//
// Created by lizhijian on 19-1-8.
//

#ifndef CUSTOM_OPS_COMMON_H
#define CUSTOM_OPS_COMMON_H

#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

void readMatImg(const char* filename, std::vector<cv::Mat>& img_t, int inputsize = 224);
void prepareImgTensor(std::vector<torch::Tensor>& img_tensor, std::vector<cv::Mat>& img_t, int index);
std::vector< std::tuple<int , float > > getInferenceResults(torch::Tensor& out_tensor);

#endif //CUSTOM_OPS_COMMON_H
