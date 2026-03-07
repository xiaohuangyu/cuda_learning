# 第八章: CUDA 性能分析 (Profiling)

本章介绍 CUDA 程序的性能分析方法，包括使用 CUDA Events 计时和 Nsight 工具套件。

## 学习目标

1. 掌握 CUDA Events 计时方法
2. 使用 Nsight Systems 进行系统级分析
3. 使用 Nsight Compute 进行内核级分析
4. 理解性能瓶颈定位方法

## 文件说明

```
08_profiling/
├── profiling_demo.cu   # 性能分析示例代码
├── CMakeLists.txt      # CMake 构建配置
└── README.md           # 本文件
```

## 编译运行

### 使用 CMake (推荐)

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make
./profiling_demo
```

### 直接使用 nvcc

```bash
nvcc -arch=sm_80 -O3 profiling_demo.cu -o profiling_demo
./profiling_demo
```

## CUDA Events 计时

CUDA Events 是 GPU 端的高精度计时器，优势包括:

- GPU 端计时，不受 CPU-GPU 同步影响
- 精度可达微秒级
- 可以测量异步操作的时间

### 基本用法

```cpp
// 创建事件
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 记录开始时间
cudaEventRecord(start);

// 执行核函数
kernel<<<grid, block>>>(args...);

// 记录结束时间
cudaEventRecord(stop);
cudaEventSynchronize(stop);

// 计算经过的时间
float ms;
cudaEventElapsedTime(&ms, start, stop);

// 清理
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

## Nsight Systems (nsys)

系统级性能分析工具，用于查看整体时间线。

### 基本用法

```bash
# 运行并收集分析数据
nsys profile --stats=true ./profiling_demo

# 查看结果
nsys-ui report.nsys-rep
```

### 常用选项

```bash
# 跟踪 CUDA API 和核函数
nsys profile --trace=cuda,nvtx ./profiling_demo

# 指定输出文件名
nsys profile -o my_profile ./profiling_demo

# 显示详细统计
nsys profile --stats=true ./profiling_demo
```

### 适用场景

- 查看整体执行时间线
- 分析 CPU-GPU 交互
- 检查内核执行顺序
- 识别空闲时间和同步等待

## Nsight Compute (ncu)

内核级性能分析工具，详细分析单个内核。

### 基本用法

```bash
# 完整分析
ncu --set full -o report ./profiling_demo

# 查看结果
ncu-ui report.ncu-rep
```

### 常用选项

```bash
# 只分析特定内核
ncu -k vector_add --set full ./profiling_demo

# 快速分析 (less overhead)
ncu --set quick ./profiling_demo

# 导出为 CSV
ncu --set full --csv -o report.csv ./profiling_demo
```

### 关键指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| Memory Throughput | 内存吞吐量 | 接近峰值带宽 |
| Compute Throughput | 计算吞吐量 | 接近峰值算力 |
| Warp Execution Efficiency | Warp 执行效率 | 接近 100% |
| Shared Memory Bank Conflicts | 共享内存 bank 冲突 | 0 |
| Global Memory Load Efficiency | 全局内存加载效率 | 接近 100% |

## Roofline 模型

Nsight Compute 提供 Roofline 分析视图:

```
性能 (GFLOPS)
    ^
    │      计算密集区
    │     /
    │    /  ← 屋顶线
    │   /
    │──/────────────────
    │ /  访存密集区
    │/
    └──────────────────→ AI (FLOP/byte)
```

- **脊点以左**: 访存密集型，优化访存
- **脊点以右**: 计算密集型，优化计算

## 常见性能问题

### 1. 内存带宽未饱和

可能原因:
- 线程数不足
- 内存访问不合并
- L1/L2 缓存利用率低

解决方案:
- 增加并行度
- 使用向量化访问 (float4)
- 优化数据布局

### 2. 计算资源未充分利用

可能原因:
- Warp 发散
- 指令延迟高
- 寄存器压力

解决方案:
- 减少分支
- 使用 __restrict__ 和 const
- 调整 block 大小

### 3. 同步开销大

可能原因:
- 过多的 cudaDeviceSynchronize
- 不必要的同步

解决方案:
- 使用 CUDA 流
- 重叠计算和传输

## 扩展阅读

- [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)