# Chapter 04: 线程层次结构 - Grid/Block/Thread

## 概述

本示例详细演示CUDA的线程层次结构，包括Grid（网格）、Block（线程块）和Thread（线程）的组织方式，以及如何使用内置变量计算线程索引。

## 示例文件

- `thread_index.cu` - 线程索引演示程序

## 线程层次结构

```
Grid (网格)
├── Block(0,0)     Block(1,0)     Block(2,0)  ...
│     │               │               │
│   Thread         Thread          Thread
│     │               │               │
│   (0,0)..(n,m)   (0,0)..(n,m)    (0,0)..(n,m)
│
├── Block(0,1)     Block(1,1)     ...
│
└── ...
```

### 三个层次

| 层次 | 说明 | 最大限制 |
|------|------|----------|
| Grid | 线程块的集合 | x/y/z: 2^31-1, 65535, 65535 |
| Block | 线程的集合 | 最多1024个线程 |
| Thread | 最小执行单元 | - |

## 内置变量

### blockIdx
- 当前线程块在网格中的索引
- 类型：uint3
- 成员：blockIdx.x, blockIdx.y, blockIdx.z

### threadIdx
- 当前线程在线程块中的索引
- 类型：uint3
- 成员：threadIdx.x, threadIdx.y, threadIdx.z

### blockDim
- 线程块的维度
- 类型：dim3
- 成员：blockDim.x, blockDim.y, blockDim.z

### gridDim
- 网格的维度
- 类型：dim3
- 成员：gridDim.x, gridDim.y, gridDim.z

## 索引计算公式

### 一维索引
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二维索引
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;
```

### 三维索引
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * height * width + y * width + x;
```

## dim3 类型

```cpp
// 一维
dim3 block(256);
dim3 grid((N + 255) / 256);

// 二维
dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);

// 三维
dim3 block(8, 8, 8);
dim3 grid((dx + 7) / 8, (dy + 7) / 8, (dz + 7) / 8);

// 核函数调用
kernel<<<grid, block>>>(args...);
```

## 编译与运行

```bash
mkdir build && cd build
cmake ..
make
./thread_index
```

## 学习要点

1. 理解Grid-Block-Thread三级层次结构
2. 掌握blockIdx、threadIdx、blockDim、gridDim内置变量
3. 学会一维、二维、三维索引的计算方法
4. 理解dim3类型的使用
5. 了解线程块大小选择的考虑因素

## 常用线程块配置

| 场景 | 推荐配置 |
|------|----------|
| 1D数组 | 256 或 512 线程 |
| 2D图像 | 16x16 或 32x32 |
| 3D体积 | 8x8x8 或 4x4x4 |

注意：每个线程块最多1024个线程！