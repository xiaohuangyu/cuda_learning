# 第 02 章示例: cuda_basics
cmake_minimum_required(VERSION 3.18)

# 项目配置 - CUDA 语言已在父 CMakeLists.txt 中启用
project(02_cuda_intro LANGUAGES CUDA CXX)

# CUDA 架构由父 CMakeLists.txt 通过 CMAKE_CUDA_ARCHITECTURES 设置
# 如需单独设置，可取消注释:
# set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architecture" FORCE)

# CUDA 编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(cuda_basics cuda_basics.cu)

# 链接 CUDA 运行时库
target_link_libraries(cuda_basics CUDA::cudart)

# 设置输出目录
set_target_properties(cuda_basics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/02_cuda_intro"
)
