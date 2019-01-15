cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_app)

SET(CMAKE_PREFIX_PATH "./libtorch")
include_directories(inc)
include_directories(src)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(example-app src/model_load.cpp inc/common.h src/utils.cpp)
target_link_libraries(example-app ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
