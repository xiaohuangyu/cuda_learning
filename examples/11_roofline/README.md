# 第十一章: Roofline 模型与性能分析

本章介绍 Roofline 模型，以及如何用它指导 CUDA 程序优化。

## 学习目标

1. 理解 Roofline 模型的含义和用途
2. 学会计算算子的计算强度 (Arithmetic Intensity)
3. 使用 Nsight Compute 进行 Roofline 分析
4. 根据瓶颈类型选择优化策略

## 文件说明

```
11_roofline/
├── roofline_demo.cu   # Roofline 模型示例代码
├── CMakeLists.txt     # CMake 构建配置
└── README.md          # 本文件
```

## 编译运行

### 使用 CMake (推荐)

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make
./roofline_demo
```

### 直接使用 nvcc

```bash
nvcc -arch=sm_80 -O3 roofline_demo.cu -o roofline_demo
./roofline_demo
```

## Roofline 模型基础

### 核心公式

```
性能 (GFLOPS) = min(峰值算力, 带宽 × 计算强度)
```

### 计算强度 (Arithmetic Intensity, AI)

```
AI = FLOP / Bytes
```

- FLOP: 浮点运算次数
- Bytes: 数据访问量 (字节)

### Roofline 图示

```
性能 (GFLOPS)
    ^
    │        计算密集区 (Compute-Bound)
    │       ┌────────────────
    │      ╱ ← 峰值算力天花板
    │     ╱
    │    ╱
    │   ╱
    │  ╱ ← 脊点 (Ridge Point)
    │ ╱
    │╱ 访存密集区 (Memory-Bound)
    └──────────────────────────→ AI
      0.1    1      10     100
```

## 不同算子的 AI 计算

### 1. 向量加法

```
计算: c[i] = a[i] + b[i]

FLOP: 1 (一次加法)
Bytes: 12 (2读 + 1写, FP32)
AI = 1/12 ≈ 0.083 FLOP/byte
```

### 2. 向量 FMA

```
计算: d[i] = a[i] * b[i] + c[i]

FLOP: 2 (乘法 + 加法)
Bytes: 16 (3读 + 1写)
AI = 2/16 = 0.125 FLOP/byte
```

### 3. 矩阵乘法 (N×N)

```
FLOP: 2N³ (N³ 次乘加)
Bytes: 3N² (2读 + 1写)
AI = 2N/3

当 N=1024: AI ≈ 682 FLOP/byte
```

## 脊点计算

脊点是 Roofline 模型的关键参数:

```
脊点 AI = 峰值算力 / 峰值带宽
```

### 常见 GPU 的脊点

| GPU | FP32 峰值 | 带宽 | FP32 脊点 |
|-----|-----------|------|-----------|
| A100 | 19.5 TFLOPS | 2039 GB/s | ~9.6 |
| V100 | 15.7 TFLOPS | 900 GB/s | ~17.4 |
| RTX 4090 | 82.6 TFLOPS | 1008 GB/s | ~82 |

## 优化策略

### 访存密集区 (AI < 脊点)

性能受限于内存带宽，优化重点在访存:

1. **向量化访问**
   ```cpp
   // 使用 float4 一次加载 4 个元素
   float4 va = a[idx];
   ```

2. **降低精度**
   ```cpp
   // FP16 数据量减半，AI 翻倍
   AI_FP32 = 1/12 = 0.083
   AI_FP16 = 1/6  = 0.167  // 翻倍!
   ```

3. **算子融合**
   ```cpp
   // 融合: elementwise_add + relu
   // 减少 1 次全局内存读写
   ```

### 计算密集区 (AI > 脊点)

性能受限于计算能力，优化重点在计算:

1. **Tensor Core 加速**
   ```cpp
   // 使用 WMMA 或 CUTLASS
   ```

2. **指令级并行**
   ```cpp
   // 循环展开
   #pragma unroll
   for (int i = 0; i < 4; i++) { ... }
   ```

3. **减少分支**
   ```cpp
   // 避免 warp 发散
   ```

## 使用 Nsight Compute 分析

```bash
# 完整分析
ncu --set full -o report ./roofline_demo

# 查看 Roofline
ncu-ui report.ncu-rep
```

### 关键指标

| 指标 | 说明 | 查看 |
|------|------|------|
| AI | 计算强度 | Memory section |
| 带宽利用率 | 实际/峰值带宽 | Memory Throughput |
| 计算利用率 | 实际/峰值算力 | Compute Throughput |

## FP16 的 Roofline 优势

```
FP16 优化在 Roofline 上的效果:

1. 数据量减半 → 向右移动 (AI 翻倍)
2. Tensor Core → 峰值算力更高
3. 可能跨越脊点 → 瓶颈类型改变!

FP32 vector_add: AI = 0.083 (访存密集)
FP16 vector_add: AI = 0.167 (仍访存密集，但更接近脊点)
```

## 代码示例

### 不同计算强度的核函数

```cpp
// AI = 0.083 (访存密集)
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 1 FLOP, 12 bytes
    }
}

// AI = 2.5 (计算密集)
__global__ void compute_intensive(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            val = val * val + b[idx];  // 大量计算
        }
        c[idx] = val;
    }
}
```

## 扩展阅读

- [Roofline Model 原始论文](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoRed.pdf)
- [Nsight Compute Roofline 分析](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#roofline)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)