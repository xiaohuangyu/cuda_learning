# 第五章：第一个 CUDA 程序 - Hello CUDA

## 学习目标

本章将带你编写第一个 CUDA 程序，学习以下内容：

1. CUDA 程序的基本结构
2. 如何编写内核函数（kernel function）
3. 如何从 CPU 调用 GPU 内核
4. 如何获取线程索引信息

## 文件说明

```
05_hello_cuda/
├── hello_cuda.cu    # 主程序源码
├── CMakeLists.txt   # CMake 配置文件
└── README.md        # 本说明文件
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
./hello_cuda
```

### 方法二：直接使用 nvcc

```bash
# 编译（根据你的 GPU 选择合适的架构）
nvcc -arch=sm_80 hello_cuda.cu -o hello_cuda

# 运行
./hello_cuda
```

## 代码核心概念

### 1. 内核函数（Kernel Function）

```cpp
__global__ void hello_cuda()
{
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}
```

- `__global__`：CUDA 函数限定符，表示这是一个内核函数
- 内核函数在 GPU 上执行，由 CPU 调用
- 返回类型必须是 `void`

### 2. 内核启动语法

```cpp
kernel_name<<<grid_size, block_size>>>(arguments...);
```

- `grid_size`：网格中的块数量
- `block_size`：每个块中的线程数量
- `<<<>>>`：执行配置语法

### 3. 线程索引变量

| 变量 | 说明 |
|------|------|
| `blockIdx.x` | 当前块在网格中的索引 |
| `threadIdx.x` | 当前线程在块内的索引 |
| `blockDim.x` | 每个块的线程数量 |
| `gridDim.x` | 网格中的块数量 |

### 4. 全局索引计算

```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

这是 CUDA 编程中最常用的索引计算公式。

## 预期输出

```
=========================================
   第五章：第一个 CUDA 程序 - Hello CUDA
=========================================

【第一部分】最简单的内核调用
-----------------------------------------
启动配置: <<<1, 1>>> (1 个块, 每块 1 个线程)
总线程数: 1

Hello from GPU! Block 0, Thread 0

内核执行完成！

【第二部分】多线程内核调用
-----------------------------------------
启动配置: <<<2, 4>>> (2 个块, 每块 4 个线程)
总线程数: 2 * 4 = 8

Hello! 全局ID:  0 (Block 0, Thread 0)
Hello! 全局ID:  1 (Block 0, Thread 1)
Hello! 全局ID:  2 (Block 0, Thread 2)
Hello! 全局ID:  3 (Block 0, Thread 3)
Hello! 全局ID:  4 (Block 1, Thread 0)
Hello! 全局ID:  5 (Block 1, Thread 1)
Hello! 全局ID:  6 (Block 1, Thread 2)
Hello! 全局ID:  7 (Block 1, Thread 3)

=========================================
           学习要点总结
=========================================
...
```

## 常见问题

### Q1: 如何选择 CUDA 架构？

根据你的 GPU 型号选择：

| GPU 型号 | 架构 |
|----------|------|
| V100 | sm_70 |
| T4, RTX 2080 | sm_75 |
| A100, RTX 3090 | sm_80 |
| RTX 4090 | sm_89 |
| H100 | sm_90 |

### Q2: 为什么需要 cudaDeviceSynchronize()？

内核启动是异步的，CPU 不会等待 GPU 执行完成。调用此函数可以：
- 确保 GPU 的 printf 输出显示在屏幕上
- 确保 GPU 完成计算后再访问结果
- 用于调试和性能测量

### Q3: printf 输出的顺序为什么是乱序的？

多个线程并行执行，printf 的输出顺序不确定。如果需要有序输出，应该在 CPU 端处理。

## 下一步

完成本章后，继续学习：
- 第六章：内存管理基础（cudaMalloc, cudaMemcpy, cudaFree）
- 第七章：内核深入理解（__global__, __device__, __host__）