cmake_minimum_required(VERSION 3.10)
project(deeplab)  # 项目名称

set(project_name deeplab)
set(project_root ${PROJECT_SOURCE_DIR})
set(project_src ${project_root}/src)
set(project_lib ${project_root}/lib)
set(project_include ${project_include}/include)
set(EXECUTABLE_OUTPUT_PATH ${project_root}/bin)


option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)
add_definitions(-pthread)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE "Release") #Debug Release
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O -Wall -g -ggdb")#在debug下不优化,即采用最低级别的O0
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")#在release下采用最高级别优化


link_directories(
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu/
        /opt/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib
)

include_directories(
        /usr/local/cuda/include
        /usr/include/x86_64-linux-gnu/
        /opt/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/include
        ${project_src}/common
        ${project_src}/segmentation
)
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)



aux_source_directory(${project_src} src)
aux_source_directory(${project_src}/common src)
aux_source_directory(${project_src}/segmentation src)

add_executable(${project_name} ${src})

target_link_libraries(${project_name} ${CMAKE_DL_LIBS}
        nvinfer
        cudart
        ${OpenCV_LIBS}
        )

