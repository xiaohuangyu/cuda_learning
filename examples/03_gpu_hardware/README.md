# Chapter 03: GPU硬件 - 设备信息查询

## 概述

本示例演示如何使用CUDA运行时API查询GPU设备的硬件属性，包括SM数量、内存大小、计算能力等关键信息。

## 示例文件

- `device_info.cu` - GPU设备属性查询程序

## cudaDeviceProp 关键字段

### 基本信息

| 字段 | 类型 | 说明 |
|------|------|------|
| name | char[256] | 设备名称 |
| major, minor | int | 计算能力版本号 |
| multiProcessorCount | int | SM数量 |

### 内存信息

| 字段 | 类型 | 说明 |
|------|------|------|
| totalGlobalMem | size_t | 全局显存总量 |
| sharedMemPerBlock | size_t | 每块共享内存 |
| sharedMemPerMultiprocessor | size_t | 每SM共享内存 |
| totalConstMem | size_t | 常量内存总量 |
| regsPerBlock | int | 每块寄存器数 |

### 线程限制

| 字段 | 类型 | 说明 |
|------|------|------|
| maxThreadsPerBlock | int | 每块最大线程数 |
| maxThreadsDim[3] | int | 块最大维度 |
| maxGridSize[3] | int | 网格最大维度 |
| maxThreadsPerMultiProcessor | int | 每SM最大线程数 |
| warpSize | int | Warp大小（通常32） |

### 性能参数

| 字段 | 类型 | 说明 |
|------|------|------|
| clockRate | int | GPU时钟频率（kHz） |
| memoryClockRate | int | 内存时钟频率（kHz） |
| memoryBusWidth | int | 内存总线宽度（bit） |
| l2CacheSize | int | L2缓存大小 |

## 核心API函数

### cudaGetDeviceCount()
```cpp
int device_count;
cudaGetDeviceCount(&device_count);
```

### cudaGetDeviceProperties()
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);
```

### cudaSetDevice()
```cpp
cudaSetDevice(device_id);  // 选择要使用的设备
```

## 编译与运行

```bash
mkdir build && cd build
cmake ..
make
./device_info
```

## 学习要点

1. 理解SM（Streaming Multiprocessor）的概念
2. 了解Warp的概念和大小（32线程）
3. 掌握计算能力的含义
4. 了解GPU内存层次结构
5. 理解硬件限制对编程的影响

## 常见GPU架构

| 架构 | 计算能力 | 代表产品 |
|------|----------|----------|
| Ampere | 8.0, 8.6 | RTX 30系列, A100 |
| Turing | 7.5 | RTX 20系列, T4 |
| Volta | 7.0 | V100, Titan V |
| Pascal | 6.0, 6.1 | GTX 10系列, P100 |
| Maxwell | 5.0, 5.2 | GTX 9系列 |
| Kepler | 3.0, 3.5 | K80, K40 |