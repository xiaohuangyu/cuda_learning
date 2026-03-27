# 第 01 章示例: cpu_vs_gpu
cmake_minimum_required(VERSION 3.18)

# 项目配置 - CUDA 语言已在父 CMakeLists.txt 中启用
project(01_gpu_basics LANGUAGES CUDA CXX)

# CUDA 架构由父 CMakeLists.txt 通过 CMAKE_CUDA_ARCHITECTURES 设置
# 如需单独设置，可取消注释:
# set(CMAKE_CUDA_ARCHIT
# ECTURES 80 CACHE STRING "CUDA architecture" FORCE)

# CUDA 编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(cpu_vs_gpu cpu_vs_gpu.cu)
add_executable(cpu_vs_gpu_1 cpu_vs_gpu_1.cu)
add_executable(01_basics_01 basics_01.cu)
add_executable(01_stream_event stream_event.cu)

# 链接 CUDA 运行时库
target_link_libraries(cpu_vs_gpu CUDA::cudart)

# 设置输出目录
set_target_properties(cpu_vs_gpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/01_gpu_basics"
)
