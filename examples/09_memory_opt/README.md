# 第九章: CUDA 内存优化

本章介绍 CUDA 内存访问优化技术，重点讲解合并访问和向量化访存。

## 学习目标

1. 理解合并访问 (Coalesced Access) 的概念
2. 掌握 float4 向量化访存技术
3. 学习共享内存优化方法
4. 对比不同访问模式的性能差异

## 文件说明

```
09_memory_opt/
├── memory_opt.cu     # 内存优化示例代码
├── CMakeLists.txt    # CMake 构建配置
└── README.md         # 本文件
```

## 编译运行

### 使用 CMake (推荐)

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make
./memory_opt
```

### 直接使用 nvcc

```bash
nvcc -arch=sm_80 -O3 memory_opt.cu -o memory_opt
./memory_opt
```

## 合并访问 (Coalesced Access)

### 概念

合并访问是指相邻线程访问连续的内存地址，GPU 可以将这些访问合并为最少的内存事务。

### 合并访问模式

```
理想情况 (合并):
Thread 0 -> addr[0]
Thread 1 -> addr[1]    <- 相邻线程访问相邻地址
Thread 2 -> addr[2]
...

非合并访问 (跨度):
Thread 0 -> addr[0]
Thread 1 -> addr[32]   <- 跨度太大，无法合并
Thread 2 -> addr[64]
...
```

### 代码示例

```cpp
// 合并访问 - 最佳实践
__global__ void coalesced_access(float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data[idx] * 2.0f;  // 连续地址
    }
}

// 非合并访问 - 应避免
__global__ void strided_access(float* data, float* out, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;  // 步长访问
    if (idx < n) {
        out[idx] = data[idx] * 2.0f;
    }
}
```

## 向量化访存

### 概念

向量化访存使用 float2/float4 等向量类型，一次读写多个数据元素。

### 优势

1. 减少内存事务次数
2. 提高内存总线利用率
3. 128 位对齐获得最佳性能

### 代码示例

```cpp
// 标量访问 - 每次读写 1 个 float
__global__ void scalar_access(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// float4 向量化 - 每次读写 4 个 float
__global__ void vectorized_float4(float4* a, float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        float4 va = a[idx];  // 一次加载 16 字节
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}
```

### 性能对比

| 访问方式 | 每次读写 | 加速比 |
|----------|----------|--------|
| float (标量) | 4 字节 | 1.0x |
| float2 (向量) | 8 字节 | ~1.2x |
| float4 (向量) | 16 字节 | ~1.4x |

## 共享内存优化

### 矩阵转置示例

```cpp
#define TILE_SIZE 32

__global__ void transpose_shared(float* input, float* output,
                                  int width, int height) {
    // 声明共享内存
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 从全局内存加载到共享内存 (合并读取)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // 计算转置位置
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // 从共享内存写入全局内存 (合并写入)
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Bank Conflict 优化

共享内存被分为 32 个 bank，每个 bank 每周期只能服务一个地址。

```cpp
// 有 bank conflict
__shared__ float tile[32][32];
// 多个线程访问 tile[0][0], tile[1][0], tile[2][0]...
// 不同行相同列 -> 同一 bank -> conflict!

// 无 bank conflict
__shared__ float tile[32][33];  // +1 填充
// 访问 tile[0][0], tile[1][1], tile[2][2]...
// 不同行不同列 -> 不同 bank -> 无 conflict!
```

## 内存访问最佳实践

### 1. 全局内存

- 优先保证合并访问
- 使用向量化类型 (float4)
- 考虑内存对齐

### 2. 共享内存

- 用于优化访问模式
- 避免 bank conflict
- 注意同步开销

### 3. 常量内存

- 适合只读且广播的数据
- 使用 `__constant__` 限定符

### 4. 纹理内存

- 适合有空间局部性的访问
- 自动缓存

## 使用 Nsight Compute 分析

```bash
# 分析内存吞吐量
ncu --set full -o report ./memory_opt

# 查看关键指标
ncu --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak ./memory_opt
```

### 关键指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| Global Load Efficiency | 全局内存加载效率 | > 80% |
| Global Store Efficiency | 全局内存存储效率 | > 80% |
| Shared Load Efficiency | 共享内存加载效率 | ~100% |
| L2 Cache Hit Rate | L2 缓存命中率 | 越高越好 |

## 扩展阅读

- [CUDA Best Practices Guide - Memory Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [CUDA Programming Guide - Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-access-to-global-memory)