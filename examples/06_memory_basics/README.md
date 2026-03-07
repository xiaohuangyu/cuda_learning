# 第六章：CUDA 内存管理基础

## 学习目标

本章将学习 CUDA 内存管理的三个核心操作：

1. `cudaMalloc` - 在 GPU 上分配内存
2. `cudaMemcpy` - 在 CPU 和 GPU 之间传输数据
3. `cudaFree` - 释放 GPU 内存

## 文件说明

```
06_memory_basics/
├── memory_demo.cu    # 主程序源码
├── CMakeLists.txt    # CMake 配置文件
└── README.md         # 本说明文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake ..

# 编译
make

# 运行
./memory_demo
```

### 方法二：直接使用 nvcc

```bash
nvcc -arch=sm_80 memory_demo.cu -o memory_demo
./memory_demo
```

## CUDA 内存管理核心概念

### 1. cudaMalloc - GPU 内存分配

```cpp
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

在 GPU 上分配全局内存。

**参数说明：**
- `devPtr`: 指向 GPU 内存指针的指针（二级指针）
- `size`: 要分配的字节数

**使用示例：**
```cpp
float *d_ptr;
cudaMalloc((void**)&d_ptr, N * sizeof(float));
```

**注意事项：**
- 分配的内存地址对齐到 256 字节边界
- 需要使用 `cudaFree` 释放
- GPU 内存有独立的地址空间

### 2. cudaMemcpy - 数据传输

```cpp
cudaError_t cudaMemcpy(void *dst, const void *src,
                       size_t count, cudaMemcpyKind kind);
```

在 CPU 和 GPU 之间传输数据。

**传输方向参数：**

| 参数值 | 方向 | 说明 |
|--------|------|------|
| `cudaMemcpyHostToDevice` | CPU -> GPU | 数据上传到 GPU |
| `cudaMemcpyDeviceToHost` | GPU -> CPU | 数据下载到 CPU |
| `cudaMemcpyDeviceToDevice` | GPU -> GPU | GPU 内部复制 |

**使用示例：**
```cpp
// CPU -> GPU
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

// GPU -> CPU
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
```

**注意事项：**
- 这是同步操作，会阻塞 CPU
- 传输速度受 PCIe 带宽限制
- 是 CUDA 程序的主要性能瓶颈之一

### 3. cudaFree - 释放内存

```cpp
cudaError_t cudaFree(void *devPtr);
```

释放 GPU 内存。

**使用示例：**
```cpp
cudaFree(d_ptr);
d_ptr = NULL;  // 建议置空
```

## 完整的 CUDA 程序流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 分配主机内存 (malloc)                                    │
│     h_a = (float*)malloc(size);                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 分配设备内存 (cudaMalloc)                                │
│     cudaMalloc(&d_a, size);                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 传输数据到 GPU (cudaMemcpy H2D)                          │
│     cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 执行内核                                                 │
│     kernel<<<grid, block>>>(d_a, ...);                     │
│     cudaDeviceSynchronize();                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 传输结果回 CPU (cudaMemcpy D2H)                          │
│     cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  6. 释放内存                                                 │
│     cudaFree(d_a);                                         │
│     free(h_a);                                             │
└─────────────────────────────────────────────────────────────┘
```

## 错误处理

使用 `CUDA_CHECK` 宏进行错误检查：

```cpp
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// 使用方法
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
```

## 预期输出

```
=========================================================
   第六章：CUDA 内存管理基础
=========================================================

数组配置:
  元素数量: 1024
  每个数组大小: 4.00 KB

【步骤 1】分配主机（CPU）内存
---------------------------------------------------------
h_a = [1.0, 1.0, 1.0, 1.0, 1.0, ...] (共 1024 个元素)
h_b = [2.0, 2.0, 2.0, 2.0, 2.0, ...] (共 1024 个元素)
主机内存分配完成

【步骤 2】分配设备（GPU）内存
---------------------------------------------------------
GPU 内存分配成功:
  d_a: 0x7f...
  d_b: 0x7f...
  d_c: 0x7f...
  总共分配: 12.00 KB

【步骤 3】传输数据到 GPU（Host -> Device）
---------------------------------------------------------
数据传输完成:
  h_a -> d_a (4096 字节)
  h_b -> d_b (4096 字节)

【步骤 4】在 GPU 上执行向量加法内核
---------------------------------------------------------
内核配置:
  每块线程数: 256
  块数量: 4
  总线程数: 1024
  实际需要的线程数: 1024

内核执行完成

【步骤 5】传输结果回 CPU（Device -> Host）
---------------------------------------------------------
h_c (结果) = [3.0, 3.0, 3.0, 3.0, 3.0, ...] (共 1024 个元素)
结果传输完成

【步骤 6】验证计算结果
---------------------------------------------------------
验证通过！所有结果正确。
  输入: a = 1.0, b = 2.0
  输出: c = a + b = 3.0

【步骤 7】释放内存
---------------------------------------------------------
GPU 内存释放完成
CPU 内存释放完成
```

## 内存管理最佳实践

1. **减少数据传输**
   - CPU-GPU 数据传输是主要性能瓶颈
   - 尽量减少传输次数
   - 传输大块数据比多次传输小块数据更高效

2. **使用统一内存**
   - 对于简单程序，考虑使用 `cudaMallocManaged`
   - 自动管理 CPU 和 GPU 之间的数据传输

3. **异步传输**
   - 使用 `cudaMemcpyAsync` 进行异步数据传输
   - 可以与计算重叠

4. **及时释放内存**
   - GPU 显存有限，及时释放不再使用的内存
   - 使用 `CUDA_CHECK` 检查释放操作

## 下一步

完成本章后，继续学习：
- 第七章：内核深入理解（__global__, __device__, __host__）