# CUDA 学习教程

> 从零开始学习 CUDA 并行编程，适合完全新手

## 项目简介

本项目是一套完整的 CUDA 学习教程，包含：

- **11 章教程文档**：循序渐进，从基础到进阶
- **11 个示例代码**：每章配套可运行代码，详细注释
- **CMake 编译支持**：一键编译所有示例
- **辅助脚本**：环境检查、编译、快速运行

## 目录结构

```
cuda_learning/
├── tutorials/                    # 教程文档
│   ├── 01_什么是GPU并行计算.md
│   ├── 02_CUDA是什么.md
│   ├── 03_GPU硬件架构入门.md
│   ├── 04_线程层级结构.md
│   ├── 05_第一个CUDA程序.md
│   ├── 06_内存管理基础.md
│   ├── 07_核函数深入.md
│   ├── 08_性能分析入门.md
│   ├── 09_内存访问优化.md
│   ├── 10_精度与性能.md
│   └── 11_Roofline模型.md
│
├── examples/                     # 示例代码（与章节对应）
│   ├── CMakeLists.txt
│   ├── 01_gpu_basics/           # 第1章：GPU基础
│   ├── 02_cuda_intro/           # 第2章：CUDA入门
│   ├── 03_gpu_hardware/         # 第3章：GPU硬件
│   ├── 04_thread_hierarchy/     # 第4章：线程层级
│   ├── 05_hello_cuda/           # 第5章：第一个程序
│   ├── 06_memory_basics/        # 第6章：内存管理
│   ├── 07_kernel_deep/          # 第7章：核函数深入
│   ├── 08_profiling/            # 第8章：性能分析
│   ├── 09_memory_opt/           # 第9章：内存优化
│   ├── 10_precision/            # 第10章：精度与性能
│   └── 11_roofline/             # 第11章：Roofline模型
│
├── scripts/                      # 辅助脚本
│   ├── check_env.sh             # 环境检查
│   ├── compile_all.sh           # 一键编译
│   └── run_example.sh           # 快速运行示例
│
└── *.pdf                         # 课程参考资料
```

## 快速开始

### 1. 环境要求

- NVIDIA GPU（Compute Capability 5.3+）
- CUDA Toolkit 11.0+
- CMake 3.18+
- Linux 操作系统

### 2. 检查环境

```bash
# 运行环境检查脚本
./scripts/check_env.sh
```

### 3. 编译示例

```bash
# 基本编译
./scripts/compile_all.sh

# 清理后重新编译
./scripts/compile_all.sh --clean

# 指定 GPU 架构
./scripts/compile_all.sh --arch=80    # A100
./scripts/compile_all.sh --arch=86    # RTX 3090
./scripts/compile_all.sh --arch=89    # RTX 4090
```

### 4. 运行示例

```bash
# 列出所有示例
./scripts/run_example.sh --list

# 运行指定章节
./scripts/run_example.sh 5    # 第5章：第一个程序
./scripts/run_example.sh 3    # 第3章：GPU信息

# 运行所有示例
./scripts/run_example.sh --all
```

## 章节内容

### 第一部分：基础入门

| 章节 | 教程 | 示例 | 学习目标 |
|:----:|------|------|----------|
| 01 | [什么是GPU并行计算](tutorials/01_什么是GPU并行计算.md) | `cpu_vs_gpu` | 理解并行计算概念，CPU vs GPU 区别 |
| 02 | [CUDA是什么](tutorials/02_CUDA是什么.md) | `cuda_basics` | 了解 CUDA 架构，Host/Device 概念 |
| 03 | [GPU硬件架构入门](tutorials/03_GPU硬件架构入门.md) | `device_info` | 认识 SM、CUDA Core、Warp |
| 04 | [线程层级结构](tutorials/04_线程层级结构.md) | `thread_index` | 掌握 Grid/Block/Thread 组织 |
| 05 | [第一个CUDA程序](tutorials/05_第一个CUDA程序.md) | `hello_cuda` | 编写并运行第一个程序 |
| 06 | [内存管理基础](tutorials/06_内存管理基础.md) | `memory_demo` | 学会 cudaMalloc/cudaMemcpy |
| 07 | [核函数深入](tutorials/07_核函数深入.md) | `kernel_demo` | 理解 __global__、<< <>>> |

### 第二部分：进阶优化

| 章节 | 教程 | 示例 | 学习目标 |
|:----:|------|------|----------|
| 08 | [性能分析入门](tutorials/08_性能分析入门.md) | `profiling_demo` | 使用 nsys/ncu 分析性能 |
| 09 | [内存访问优化](tutorials/09_内存访问优化.md) | `memory_opt` | 合并访问、向量化访存 |
| 10 | [精度与性能](tutorials/10_精度与性能.md) | `precision_demo` | FP16 编程与优化 |
| 11 | [Roofline模型](tutorials/11_Roofline模型.md) | `roofline_demo` | 分析性能瓶颈 |

## GPU 架构参考

| 架构 | GPU 示例 | Compute Capability | CMake 参数 |
|------|----------|-------------------|------------|
| Volta | V100 | 7.0 | `-DCMAKE_CUDA_ARCHITECTURES=70` |
| Turing | RTX 2080, T4 | 7.5 | `-DCMAKE_CUDA_ARCHITECTURES=75` |
| Ampere | A100 | 8.0 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| Ampere | RTX 3090, A40 | 8.6 | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| Ada Lovelace | RTX 4090 | 8.9 | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| Hopper | H100 | 9.0 | `-DCMAKE_CUDA_ARCHITECTURES=90` |

## 手动编译

如果不想使用脚本，可以手动编译：

```bash
cd examples
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j

# 运行示例
./05_hello_cuda/hello_cuda
```

## 教程特点

- **从零开始**：假设读者完全不了解 GPU 编程
- **图解优先**：使用 Mermaid 图解释所有抽象概念
- **代码说话**：每行代码都有详细中文注释
- **即学即练**：每个概念都配可运行代码
- **循序渐进**：每章只引入 2-3 个新概念

## 学习路线

```
第一部分：基础入门
    │
    ├── 第1章：理解 GPU 并行计算原理
    │
    ├── 第2章：了解 CUDA 基本概念
    │
    ├── 第3章：认识 GPU 硬件架构
    │
    ├── 第4章：掌握线程组织方式
    │
    ├── 第5章：编写第一个程序
    │
    ├── 第6章：学会内存管理
    │
    └── 第7章：深入理解核函数
           │
           ▼
第二部分：进阶优化
    │
    ├── 第8章：学会性能分析
    │
    ├── 第9章：优化内存访问
    │
    ├── 第10章：使用低精度加速
    │
    └── 第11章：理解 Roofline 模型
```

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA FP16 API](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html)

## 课程来源

本教程基于 InfiniTensor 大模型与人工智能系统训练营 2025 冬季课程内容整理。

---

*祝学习愉快！*