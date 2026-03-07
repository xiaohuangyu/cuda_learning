# 第十章: CUDA 精度与性能

本章介绍 CUDA 半精度 (FP16) 编程，以及如何利用精度优化来提升性能。

## 学习目标

1. 掌握 CUDA FP16 编程基础
2. 理解 half 和 half2 类型的使用
3. 学习 __hadd 和 __hadd2 等内建函数
4. 对比 FP32 与 FP16 的性能差异

## 文件说明

```
10_precision/
├── precision_demo.cu   # 精度与性能示例代码
├── CMakeLists.txt      # CMake 构建配置
└── README.md           # 本文件
```

## 编译运行

### 使用 CMake (推荐)

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make
./precision_demo
```

### 直接使用 nvcc

```bash
nvcc -arch=sm_80 -O3 precision_demo.cu -o precision_demo
./precision_demo
```

## 数据类型对比

### FP32 (单精度)

- 位宽: 32 位
- 符号位: 1 位
- 指数位: 8 位
- 尾数位: 23 位
- 有效数字: 约 7 位
- 范围: 约 1.2e-38 到 3.4e38

### FP16 (半精度)

- 位宽: 16 位
- 符号位: 1 位
- 指数位: 5 位
- 尾数位: 10 位
- 有效数字: 约 3-4 位
- 范围: 约 6.1e-5 到 6.5e4

## FP16 编程

### 基本类型

```cpp
#include <cuda_fp16.h>

// 标量类型
half h = __float2half(1.0f);
float f = __half2float(h);

// 向量类型 (2 个 half)
half2 h2 = __halves2half2(h1, h2);
```

### 类型转换

```cpp
// FP32 -> FP16
half h = __float2half(3.14f);

// FP16 -> FP32
float f = __half2float(h);

// 批量转换 (half2)
half2 h2 = __float22half2_rn(float2_val);
float2 f2 = __half22float2(h2);
```

### 运算函数

#### 标量运算

```cpp
half a, b, c;

// 加法
c = __hadd(a, b);

// 减法
c = __hsub(a, b);

// 乘法
c = __hmul(a, b);

// 除法
c = __hdiv(a, b);

// FMA: a * b + c
half result = __hfma(a, b, c);
```

#### 向量运算 (half2)

```cpp
half2 a, b, c;

// 向量加法 (SIMD)
c = __hadd2(a, b);  // c.x = a.x + b.x, c.y = a.y + b.y

// 向量乘法
c = __hmul2(a, b);

// 向量 FMA
c = __hfma2(a, b, c);
```

## 性能优势

### 1. 访存量减半

```
FP32: 16M 元素 = 64 MB
FP16: 16M 元素 = 32 MB (减半!)
```

### 2. 计算强度 (AI) 翻倍

```
向量加法:
  FP32: AI = 1 FLOP / 12 bytes = 0.083
  FP16: AI = 1 FLOP / 6 bytes  = 0.167 (翻倍!)
```

### 3. SIMD 加速

```cpp
// 标量: 每个 warp 处理 32 个元素
c[idx] = __hadd(a[idx], b[idx]);

// SIMD (half2): 每个 warp 处理 64 个元素
c[idx] = __hadd2(a[idx], b[idx]);  // 快 2 倍!
```

## 代码示例

### FP32 vs FP16 核函数对比

```cpp
// FP32 标量版本
__global__ void add_fp32(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// FP16 half2 SIMD 版本
__global__ void add_fp16_vec2(half2* a, half2* b, half2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        // 一条指令处理 2 个 half
        c[idx] = __hadd2(a[idx], b[idx]);
    }
}
```

## 精度权衡

### 精度损失示例

```
原始值 (FP32):  3.14159265
FP16 表示:      3.140625
绝对误差:       0.00096765
相对误差:       0.03%
```

### 适用场景

**适合 FP16:**
- 深度学习推理/训练
- 图形渲染
- 信号处理 (部分场景)
- 对精度要求不高的计算

**不适合 FP16:**
- 科学计算
- 金融计算
- 需要高精度的场景

## 混合精度编程

在需要精度的场景，可以使用混合精度:

```cpp
// 存储用 FP16，计算用 FP32
__global__ void mixed_precision(half* a, half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 转换为 FP32 计算
        float fa = __half2float(a[idx]);
        float fb = __half2float(b[idx]);
        float fc = fa + fb;

        // 结果存为 FP16
        c[idx] = __float2half(fc);
    }
}
```

## 架构要求

| 功能 | 最低架构 |
|------|----------|
| FP16 存储 | 所有架构 |
| FP32 计算 + FP16 存储 | SM 5.3+ |
| FP16 计算 (__hadd) | SM 5.3+ |
| FP16 SIMD (__hadd2) | SM 7.0+ |
| Tensor Core FP16 | SM 7.0+ |

## 扩展阅读

- [CUDA FP16 API 文档](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html)
- [CUDA Programming Guide - Half Precision](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#half-precision-arithmetic)
- [Tensor Core 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)