# CUDA 学习教程 - 示例代码

本目录包含 CUDA 学习教程的配套示例代码，涵盖从 GPU 基础到高级性能优化的完整学习路径。

## 目录结构

```
examples/
├── 01_gpu_basics/         # 第 1 章: GPU 基础知识
├── 02_cuda_intro/         # 第 2 章: CUDA 编程入门
├── 03_gpu_hardware/       # 第 3 章: GPU 硬件架构
├── 04_thread_hierarchy/   # 第 4 章: 线程层次结构
├── 05_hello_cuda/         # 第 5 章: Hello CUDA
├── 06_memory_basics/      # 第 6 章: 内存模型基础
├── 07_kernel_deep/        # 第 7 章: 核函数深入
├── 08_profiling/          # 第 8 章: 性能分析
├── 09_memory_opt/         # 第 9 章: 内存优化
├── 10_precision/          # 第 10 章: 精度与数值
└── 11_roofline/           # 第 11 章: Roofline 模型
```

## 章节与示例对应关系

| 章节 | 示例目录 | 内容描述 |
|------|----------|----------|
| 第 1 章 | 01_gpu_basics | GPU 基础知识：GPU 架构概述、CPU vs GPU 对比、并行计算概念 |
| 第 2 章 | 02_cuda_intro | CUDA 编程入门：CUDA 安装、nvcc 编译器、第一个 CUDA 程序 |
| 第 3 章 | 03_gpu_hardware | GPU 硬件架构：SM 结构、Warp、CUDA Core、内存层次 |
| 第 4 章 | 04_thread_hierarchy | 线程层次结构：Grid、Block、Thread 组织方式 |
| 第 5 章 | 05_hello_cuda | Hello CUDA：完整示例程序，核函数编写与调用 |
| 第 6 章 | 06_memory_basics | 内存模型基础：全局内存、共享内存、寄存器、常量内存 |
| 第 7 章 | 07_kernel_deep | 核函数深入：核函数优化技巧、同步机制、错误处理 |
| 第 8 章 | 08_profiling | 性能分析：Nsight Systems、Nsight Compute 使用方法 |
| 第 9 章 | 09_memory_opt | 内存优化：合并访问、共享内存优化、内存带宽优化 |
| 第 10 章 | 10_precision | 精度与数值：FP32/FP16/BF16 混合精度计算 |
| 第 11 章 | 11_roofline | Roofline 模型：性能上界分析、算力与带宽权衡 |

## 构建方法

### 环境要求

- NVIDIA GPU (Compute Capability 5.3+)
- CUDA Toolkit 11.0+
- CMake 3.18+
- Linux 操作系统

### 检查环境

```bash
# 检查 GPU
nvidia-smi

# 检查 CUDA 编译器
nvcc --version

# 查询 GPU Compute Capability
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 检查 CMake 版本
cmake --version
```

### 使用 CMake 构建

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目 (根据你的 GPU 架构设置)
# 常见架构: A100=80, RTX 3090=86, RTX 4090=89, H100=90
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80

# 编译所有示例
make -j

# 编译单个示例
make 05_hello_cuda
```

### 常见 GPU 架构对照表

| GPU 型号 | Compute Capability | CMAKE_CUDA_ARCHITECTURES |
|----------|-------------------|--------------------------|
| RTX 2080/T4 | 7.5 | 75 |
| A100 | 8.0 | 80 |
| RTX 3090/A40 | 8.6 | 86 |
| RTX 4090 | 8.9 | 89 |
| H100 | 9.0 | 90 |

## 学习路径

```
01_gpu_basics (GPU 基础)
        │
        │ 理解: GPU 架构、并行计算概念
        ↓
02_cuda_intro (CUDA 入门)
        │
        │ 学习: CUDA 安装、nvcc 编译
        ↓
03_gpu_hardware (硬件架构)
        │
        │ 理解: SM、Warp、内存层次
        ↓
04_thread_hierarchy (线程层次)
        │
        │ 掌握: Grid、Block、Thread 组织
        ↓
05_hello_cuda (第一个程序)
        │
        │ 实践: 核函数编写与调用
        ↓
06_memory_basics (内存基础)
        │
        │ 学习: 各类内存类型及使用
        ↓
07_kernel_deep (核函数深入)
        │
        │ 掌握: 核函数优化技巧
        ↓
08_profiling (性能分析)
        │
        │ 学会: 使用性能分析工具
        ↓
09_memory_opt (内存优化)
        │
        │ 实践: 内存访问优化技术
        ↓
10_precision (精度优化)
        │
        │ 学习: 混合精度计算
        ↓
11_roofline (Roofline 模型)
        │
        │ 掌握: 性能上界分析方法
        ↓
     进阶优化
```

## 常用命令速查

### 编译

```bash
# 使用 nvcc 直接编译
nvcc -arch=sm_80 -O3 kernel.cu -o kernel

# 启用调试信息
nvcc -g -G kernel.cu -o kernel_debug

# 指定 GCC 版本 (解决兼容性问题)
nvcc -ccbin g++-9 -arch=sm_80 kernel.cu -o kernel
```

### 性能分析

```bash
# 系统级分析 (Nsight Systems)
nsys profile --stats=true ./kernel

# 内核级分析 (Nsight Compute)
ncu --set full -o report ./kernel
ncu-ui report.ncu-rep
```

### 调试

```bash
# 使用 cuda-gdb 调试
cuda-gdb ./kernel_debug

# 使用 compute-sanitizer 检查内存错误
compute-sanitizer ./kernel
```

## 常见问题

### 1. 编译报错 "unsupported GNU version"

解决方案：指定 GCC 版本
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_HOST_COMPILER=g++-9
```

### 2. 运行报错 "invalid device ordinal"

解决方案：检查 GPU 是否存在
```bash
nvidia-smi
```

### 3. 性能比预期低

可能原因：
- 线程数不够（尝试增加 grid/block）
- 内存访问不连续（检查合并访问）
- 数据传输时间占主导（减少 Host↔Device 传输）

## 扩展资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA FP16 API](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html)