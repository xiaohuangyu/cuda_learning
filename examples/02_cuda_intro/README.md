# Chapter 02: CUDA入门 - 函数类型与Host/Device概念

## 概述

本示例详细介绍CUDA编程中的三种函数类型，以及Host（主机）和Device（设备）的核心概念。

## 示例文件

- `cuda_basics.cu` - CUDA函数类型演示程序

## CUDA函数类型

### 1. `__global__` 核函数

```cpp
__global__ void my_kernel(int *data) {
    // 核函数代码
}
```

- 在GPU上执行，由CPU调用
- 必须返回`void`
- 使用`<<<>>>`语法启动
- 每个线程执行一份代码

### 2. `__device__` 设备函数

```cpp
__device__ float my_device_func(float x) {
    return x * x;
}
```

- 在GPU上执行
- 只能被`__global__`或`__device__`函数调用
- 不能被CPU直接调用
- 用于封装GPU代码逻辑

### 3. `__host__` 主机函数

```cpp
__host__ void my_host_func() {
    // 普通C++函数
}
```

- 在CPU上执行（默认类型）
- 可以调用核函数
- 不能调用设备函数

## Host/Device 概念

### Host（主机）

| 组件 | 说明 |
|------|------|
| CPU | 中央处理器 |
| 主机内存 | 系统RAM |
| 执行 | __host__函数 |

### Device（设备）

| 组件 | 说明 |
|------|------|
| GPU | 图形处理器 |
| 设备内存 | 显存VRAM |
| 执行 | __global__/__device__函数 |

### 典型CUDA程序流程

```
1. cudaMalloc()     → 分配GPU内存
2. cudaMemcpy()     → 拷贝数据到GPU (Host→Device)
3. kernel<<<>>>()   → 启动核函数
4. cudaDeviceSynchronize() → 等待GPU完成
5. cudaMemcpy()     → 拷贝结果回CPU (Device→Host)
6. cudaFree()       → 释放GPU内存
```

## 编译与运行

```bash
mkdir build && cd build
cmake ..
make
./cuda_basics
```

## 学习要点

1. 理解三种函数类型的区别和使用场景
2. 掌握Host和Device的概念
3. 学会设备函数的模块化编程
4. 理解CUDA程序的执行流程

## 常见错误

1. 从CPU直接调用`__device__`函数 → 编译错误
2. 从`__device__`函数调用`__host__`函数 → 编译错误
3. 核函数返回非void类型 → 编译错误