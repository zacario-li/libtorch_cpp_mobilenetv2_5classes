//
// Created by lizhijian on 19-1-8.
//
#include "common.h"

//读取图片，并做一定格式预处理
//param: filename 文件名, image mat格式的vector, inputsize 请根据模型的input size进行设置，默认值为224x224
void readMatImg(const char* filename, std::vector<cv::Mat>& image, int inputsize)
{
    cv::Mat image_temp = cv::imread(filename,1);
    cv::cvtColor(image_temp,image_temp,cv::COLOR_BGR2RGB);
    cv::Mat image_float;
	//下面的步骤，是把mat里面的每个值转换到0.0~1.0这个区间，方便后面计算
    image_temp.convertTo(image_float,CV_32F,1.0/255);
	//对mat进行resize，以便符合后续net的输入尺寸
    cv::resize(image_float,image_float,cv::Size(inputsize,inputsize));
    image.push_back(image_float);
}

//把mat全部转换成tensor
//param: img_tensor 所有转换好的结果，都会保存到此处, img_t mat格式的vector, index 由于我们输入的是vector<cv::Mat>，所以我们需要指定具体的index进行tensor的转换
void prepareImgTensor(std::vector<torch::Tensor>& img_tensor, std::vector<cv::Mat>& img_t, int index)
{
    //创建cpu类型的tensor,并把mat的data部分赋值进去
	auto temp_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_t[index].data,{1,224,224,3});

	//上述步骤，tensor的排列是{1,224,224,3}，是batch,w,h,channel
	//我们要重新排列成{1,3,224,224}，即batch,channel,w,h
    temp_tensor = temp_tensor.permute({0,3,1,2});
	//下面步骤对RGB三个通道进行mean/归一化
    temp_tensor[0][0] = temp_tensor[0][0].sub_(0.485).div_(0.229);
    temp_tensor[0][1] = temp_tensor[0][1].sub_(0.456).div_(0.224);
    temp_tensor[0][2] = temp_tensor[0][2].sub_(0.406).div_(0.225);

    img_tensor.push_back(temp_tensor);
}

//对net运算的结果进行softmax，然后把对应的index和confidence取出，并存放在tuple中
std::vector<std::tuple<int, float>> getInferenceResults(torch::Tensor& out_tensor)
{
    //prepare the return object
    std::vector<std::tuple<int,float>> result_vec;
    //get the input tensor's batch number
	//我们有时候使用多张图片进行运算，所以此时我们要看一下out_tensor的size
    int num_of_results = out_tensor.sizes().size();

	//我们对out_tensor的排列，做了一次由大到小的排列，以便我们方便获取想要的值，当然，最大值，此时是排在第一位的
    std::tuple<torch::Tensor, torch::Tensor> result = out_tensor.sort(-1,true);
	
	//下面主要是一些std::tuple的操作，查阅c++11标准即可
    for (int i = 0; i < num_of_results; i++)
    {
        torch::Tensor top_scores = std::get<0>(result)[i];
        torch::Tensor top_idxs = std::get<1>(result)[i].toType(torch::kInt32);

        auto top_scores_a = top_scores.accessor<float,1>();
        auto top_idxs_a = top_idxs.accessor<int,1>();

        std::tuple<int,float> idx_conf;
        int idx = top_idxs_a[0];
        float conf = top_scores_a[0];
        idx_conf = std::make_tuple(idx,conf);

        result_vec.push_back(idx_conf);
    }
    return result_vec;
};
