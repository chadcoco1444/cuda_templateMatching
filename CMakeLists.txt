cmake_minimum_required(VERSION 2.8)
project( TemplateMatching )
find_package( OpenCV )
find_package(CUDA REQUIRED)
set(CUDA_64_BIT_DEVICE_CODE ON)

cuda_add_executable( GPU_TemplateMatching GPU_TemplateMatching.cu)
target_link_libraries(GPU_TemplateMatching ${OpenCV_LIBS} )