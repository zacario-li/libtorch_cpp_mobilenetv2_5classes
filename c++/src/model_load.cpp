#include "common.h"

int main(int argc, const char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <img1.jpg> <img2.jpg>\n";
    return -1;
  }

  //define cpu device for host processing data
  //创建一个cpu设备对象，后续有用
  torch::Device cpu_device(torch::kCPU);

  //detect GPU
  //查询gpu，并创建auto_device变量，用于gpu/cpu自动选择
  torch::DeviceType device_type;
  bool is_gpu = torch::cuda::is_available();

  if(is_gpu)
  {
    std::cout<<"gpu"<<std::endl;
	  device_type = torch::kCUDA;
  }else
  {
	  device_type = torch::kCPU;
  }
  torch::Device auto_device(device_type);
  
  // Deserialize the ScriptModule from a file using torch::jit::load().
  //反序列化pytorch模型，并加载到module变量
  torch::jit::script::Module module = torch::jit::load(argv[1],auto_device);
  
  //start to do inference
  //prepare inputs date var
  //网络的inputs，这里是个固定类型的变量
  std::vector<torch::jit::IValue> inputs;

  //try to read a image to run the model
  //make the sequence in batch,channel,w,h
  //此处我们要使用opencv读取两张图片
  std::vector<cv::Mat> image_vec; //这里，我们会把读取到的两张图片放进这个vector中，方便后续调用
  std::vector<cv::Mat>& r_image_vec = image_vec;//上面vector的引用，方便当作函数参数使用
  std::vector<torch::Tensor> img_t;//我们要把opencv读取的mat转换到libtorch的tensor变量，这里创建vector，方便后续使用
  std::vector<torch::Tensor>& r_img_t = img_t;//tensor vector的引用，方便当作函数参数使用

  //读取图片，并放入image_vec中，方便后续使用
  readMatImg(argv[2],r_image_vec);
  //从image_vec中提取图片，并转换成tensor格式，存放入img_t的vector中，以便后面步骤使用
  prepareImgTensor(r_img_t,r_image_vec,0);

  //重复上面步骤，读取第二张图片。因为我们后面要演示batch的input，所以使用了两张图片。如果你只有一张图片，没关系，vector中放入一个tensor就行了
  readMatImg(argv[3],r_image_vec);
  prepareImgTensor(r_img_t,r_image_vec,1);

  //move to device
  //把tensor放入到device中，这里auto_device会根据系统的实际情况，选择是否放入gpu
  img_t[0] = img_t[0].to(auto_device);
  img_t[1] = img_t[1].to(auto_device);

  //我们是inference，所以要把tensor的autograd关掉，不然浪费显存资源
  auto img_var = torch::autograd::make_variable(img_t[0],false);
  auto img_var1 = torch::autograd::make_variable(img_t[1],false);

  //上面我们提到过，我们使用了两张图，所以呢，我们要把两个tensor拼接成如下形式
  // [[1,3,224,224],
  //  [1,3,224,224]]
  //这样系统就能并行处理两张图片了
  auto img_vec = torch::cat({img_var,img_var1},0);
  //cat后的tensor，要push到inputs的变量中
  inputs.push_back(img_vec);
  std::cout<<"inputs:"<<inputs.size()<<std::endl;

  std::cout<<"start..."<<std::endl;
  //开始调用模型进行inference，并获取结果out_tensor
  torch::Tensor out_tensor = module.forward(inputs).toTensor();
  //std::cout<<"end..."<<"out tensor size:"<<out_tensor.sizes()<<std::endl;
  //由于此前我们用pytorch导出模型的时候，最后一层模型的输出是log_softmax，所以我们这里要转一下，变成softmax
  out_tensor = torch::softmax(out_tensor,1);
  //记得把结果提取到cpu端来处理后续步骤
  out_tensor = out_tensor.to(cpu_device);
  //我们来定一个out_tensor的引用，给后面处理推理结果使用
  torch::Tensor& r_out_tensor = out_tensor;
  //我们要从out_tensor里面取出我们要的分类index和confidence,详情请参看getInferenceResults的函数说明
  std::vector<std::tuple<int, float>> ret = getInferenceResults(r_out_tensor);

  for (auto& r : ret)
  {
      std::cout<<"id:"<<std::get<0>(r)<<" conf:"<<std::get<1>(r)<<std::endl;
  }

  return 0;
  
}