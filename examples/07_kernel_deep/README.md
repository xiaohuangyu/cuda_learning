# 第七章：CUDA 内核深入理解

## 学习目标

本章将深入理解 CUDA 内核机制：

1. 函数限定符：`__global__`, `__device__`, `__host__`
2. 内核启动配置：`<<<grid, block>>>`
3. 一维、二维、三维网格与块配置
4. 设备函数的调用

## 文件说明

```
07_kernel_deep/
├── kernel_demo.cu    # 主程序源码
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
./kernel_demo
```

### 方法二：直接使用 nvcc

```bash
nvcc -arch=sm_80 kernel_demo.cu -o kernel_demo
./kernel_demo
```

## CUDA 函数限定符

### 函数限定符对照表

| 限定符 | 执行位置 | 调用位置 | 说明 |
|--------|----------|----------|------|
| `__global__` | GPU | CPU | 内核函数 |
| `__device__` | GPU | GPU | 设备函数 |
| `__host__` | CPU | CPU | 主机函数（默认） |
| `__host__ __device__` | CPU/GPU | CPU/GPU | 可在两边调用 |

### __global__ - 内核函数

```cpp
__global__ void kernel_function()
{
    // 在 GPU 上执行
    // 由 CPU 调用
    printf("Hello from GPU!\n");
}

// 调用方式
kernel_function<<<grid, block>>>();
```

**特点：**
- 在 GPU 上执行
- 由 CPU 调用（使用 `<<<>>>` 语法）
- 返回类型必须是 `void`
- 每个线程独立执行

### __device__ - 设备函数

```cpp
__device__ int add(int a, int b)
{
    // 在 GPU 上执行
    // 只能被 GPU 函数调用
    return a + b;
}

__global__ void kernel()
{
    int result = add(1, 2);  // 调用设备函数
}
```

**特点：**
- 在 GPU 上执行
- 只能被 `__global__` 或其他 `__device__` 函数调用
- 用于封装 GPU 端的通用逻辑
- 内联函数，调用开销很小

### __host__ - 主机函数

```cpp
__host__ void cpu_function()
{
    // 在 CPU 上执行（默认行为）
    printf("Hello from CPU!\n");
}

// 等同于
void cpu_function()
{
    printf("Hello from CPU!\n");
}
```

### __host__ __device__ - 混合限定符

```cpp
__host__ __device__ int square(int x)
{
    return x * x;
}

// 在 CPU 上调用
int cpu_result = square(5);

// 在 GPU 上调用
__global__ void kernel()
{
    int gpu_result = square(threadIdx.x);
}
```

## 内核启动配置

### 一维配置

```cpp
// 语法
kernel<<<grid_size, block_size>>>(args);

// 示例
kernel<<<10, 256>>>();  // 10 个块，每块 256 个线程
```

### 二维配置

```cpp
// 语法
dim3 grid(width, height);
dim3 block(width, height);
kernel<<<grid, block>>>(args);

// 示例
dim3 grid(16, 16);    // 16x16 = 256 个块
dim3 block(32, 32);   // 32x32 = 1024 个线程/块
kernel<<<grid, block>>>();
```

### 三维配置

```cpp
// 语法
dim3 grid(x, y, z);
dim3 block(x, y, z);
kernel<<<grid, block>>>(args);

// 示例
dim3 grid(4, 4, 4);   // 4x4x4 = 64 个块
dim3 block(8, 8, 8);  // 8x8x8 = 512 个线程/块
kernel<<<grid, block>>>();
```

## 线程索引计算

### 一维索引

```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二维索引

```cpp
int global_x = blockIdx.x * blockDim.x + threadIdx.x;
int global_y = blockIdx.y * blockDim.y + threadIdx.y;

// 一维化索引（行优先）
int global_id = global_y * (gridDim.x * blockDim.x) + global_x;
```

### 三维索引

```cpp
int global_x = blockIdx.x * blockDim.x + threadIdx.x;
int global_y = blockIdx.y * blockDim.y + threadIdx.y;
int global_z = blockIdx.z * blockDim.z + threadIdx.z;

// 一维化索引
int width = gridDim.x * blockDim.x;
int height = gridDim.y * blockDim.y;
int global_id = global_z * width * height + global_y * width + global_x;
```

## 硬件限制

| 限制项 | 最大值 |
|--------|--------|
| 每块最大线程数 | 1024 |
| 块的 x 维度最大值 | 1024 |
| 块的 y 维度最大值 | 1024 |
| 块的 z 维度最大值 | 64 |
| 网格的 x 维度最大值 | 2^31 - 1 |
| 网格的 y 维度最大值 | 65535 |
| 网格的 z 维度最大值 | 65535 |

**注意：** `blockDim.x * blockDim.y * blockDim.z <= 1024`

## 预期输出

```
=========================================================
   第七章：CUDA 内核深入理解
=========================================================

=========================================
   第一部分：__device__ 函数演示
=========================================

__device__ 函数特点:
  - 在 GPU 上执行
  - 只能被 GPU 函数调用
  - 用于封装 GPU 端的通用逻辑

启动配置: <<<2, 4>>> (2 个块, 每块 4 个线程)

[demo_device_function] 线程 0: blockIdx=0, blockDim=4, threadIdx=0
[demo_device_function] 线程 1: blockIdx=0, blockDim=4, threadIdx=1
...

演示 __device__ 函数链式调用（阶乘计算）:
启动配置: <<<1, 6>>> (1 个块, 6 个线程)

线程 0: 0! = 1
线程 1: 1! = 1
线程 2: 2! = 2
线程 3: 3! = 6
线程 4: 4! = 24
线程 5: 5! = 120

=========================================
   第二部分：一维网格配置演示
=========================================
...

=========================================
   第三部分：二维网格配置演示
=========================================
...

=========================================
   第四部分：三维网格配置演示
=========================================
...

=========================================
   第五部分：__host__ __device__ 混合限定符演示
=========================================
...

=========================================
   学习要点总结
=========================================

1. 函数限定符:
   - __global__: 内核函数，GPU 执行，CPU 调用
   - __device__: 设备函数，GPU 执行，GPU 调用
   - __host__: 主机函数，CPU 执行，CPU 调用（默认）
   - __host__ __device__: 可在两边调用

2. 内核启动配置:
   一维: kernel<<<grid, block>>>(args)
   二维: kernel<<<dim3(gx,gy), dim3(bx,by)>>>(args)
   三维: kernel<<<dim3(gx,gy,gz), dim3(bx,by,bz)>>>(args)

...
```

## 常见问题

### Q1: 什么时候使用 __device__ 函数？

- 当需要在多个内核中复用代码时
- 当内核函数太长，需要拆分时
- 当需要封装复杂计算逻辑时

### Q2: 如何选择网格和块的大小？

- 每块线程数应为 32 的倍数（warp 大小）
- 通常使用 128、256、512 等
- 块的数量应足够大以充分利用 GPU

### Q3: 为什么需要 cudaDeviceSynchronize()？

- 内核启动是异步的
- 需要等待 GPU 完成才能看到 printf 输出
- 需要等待 GPU 完成才能访问结果数据

## 下一步

完成本章后，建议学习：
- 线程同步与共享内存
- 性能优化技术
- 错误处理与调试