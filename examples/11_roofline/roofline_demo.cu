/**
 * 第十一章: Roofline 模型与性能分析
 *
 * 学习目标:
 *   1. 理解 Roofline 模型的含义和用途
 *   2. 学会计算算子的计算强度 (Arithmetic Intensity)
 *   3. 使用 Nsight Compute 进行 Roofline 分析
 *   4. 对比不同算子的性能瓶颈
 *
 * 编译方法:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
 *   make
 *
 * 运行:
 *   ./roofline_demo
 *
 * 使用 Nsight Compute 分析:
 *   ncu --set full -o report ./roofline_demo
 *   ncu-ui report.ncu-rep
 *
 * Roofline 模型说明:
 *   性能 (GFLOPS) = min(峰值算力, 带宽 × 计算强度)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== 错误检查宏 ====================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA 错误 at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// ==================== 不同计算强度的核函数 ====================

/**
 * 算子 1: 向量加法 - 极低计算强度
 *
 * 计算强度 (Arithmetic Intensity, AI) 计算:
 *   FLOP (浮点运算次数): 1 次加法/元素
 *   Bytes (数据访问量): 2 读 + 1 写 = 12 bytes/元素
 *   AI = FLOP / Bytes = 1 / 12 ≈ 0.083 FLOP/byte
 *
 * 特点:
 *   - 典型的访存密集型 (Memory-Bound)
 *   - 性能受限于内存带宽
 *   - 计算资源大量空闲
 */
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 1 FLOP, 12 bytes
    }
}

/**
 * 算子 2: 向量 FMA (Fused Multiply-Add) - 低计算强度
 *
 * AI 计算:
 *   FLOP: 2 次 (1 乘 + 1 加)/元素
 *   Bytes: 3 读 + 1 写 = 16 bytes/元素
 *   AI = 2 / 16 = 0.125 FLOP/byte
 *
 * 特点:
 *   - 仍然是访存密集型
 *   - 比 vector_add 稍高的计算强度
 */
__global__ void vector_fma(float* a, float* b, float* c, float* d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = a[idx] * b[idx] + c[idx];  // 2 FLOPs, 16 bytes
    }
}

/**
 * 算子 3: 多次运算 - 中等计算强度
 *
 * AI 计算:
 *   FLOP: 10 次/元素 (多次计算)
 *   Bytes: 2 读 + 1 写 = 12 bytes/元素
 *   AI = 10 / 12 ≈ 0.833 FLOP/byte
 *
 * 特点:
 *   - 开始向计算密集型过渡
 *   - 需要根据具体 GPU 判断瓶颈
 */
__global__ void compute_medium(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];

        // 多次计算，增加计算强度
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            val = val * val + b[idx];  // 每次 2 FLOPs
        }

        c[idx] = val;  // 总计 8 FLOPs
    }
}

/**
 * 算子 4: 高计算强度 - 计算密集型
 *
 * AI 计算:
 *   FLOP: 20 次/元素
 *   Bytes: 2 读 + 1 写 = 12 bytes/元素
 *   AI = 20 / 12 ≈ 1.67 FLOP/byte
 *
 * 特点:
 *   - 计算密集型 (Compute-Bound)
 *   - 性能受限于计算能力
 *   - 内存带宽有余量
 */
__global__ void compute_intensive(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        float sum = 0.0f;

        // 大量计算
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            val = val * val + b[idx];  // 每次 2 FLOPs
            sum += val;                 // 1 FLOP
        }

        c[idx] = sum;  // 总计 30 FLOPs
    }
}

/**
 * 算子 5: 使用 float4 向量化的加法
 *
 * AI 不变，但带宽利用率提高:
 *   AI = 1 / 12 ≈ 0.083 FLOP/byte (同 vector_add)
 *   但由于向量化，实际带宽效率更高
 */
__global__ void vector_add_float4(float4* a, float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}

/**
 * 算子 6: 使用 half2 的加法 - FP16 优化
 *
 * AI 计算 (FP16):
 *   FLOP: 1 次/元素
 *   Bytes: 2 读 + 1 写 = 6 bytes/元素 (FP16 是 FP32 的一半)
 *   AI = 1 / 6 ≈ 0.167 FLOP/byte
 *
 * 特点:
 *   - FP16 的 AI 是 FP32 的 2 倍!
 *   - 在 Roofline 图上向右移动
 *   - 更接近计算密集区
 */
__global__ void vector_add_half2(half2* a, half2* b, half2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        c[idx] = __hadd2(a[idx], b[idx]);
    }
}

/**
 * 算子 7: 矩阵乘法 - 高计算强度 (简化版)
 *
 * AI 计算 (N x N 矩阵):
 *   FLOP: 2N³ (N³ 次乘加)
 *   Bytes: 3N² (2 读 + 1 写)
 *   AI = 2N³ / 3N² = 2N/3
 *   当 N = 1024 时, AI ≈ 682 FLOP/byte
 *
 * 特点:
 *   - 极高的计算强度
 *   - 典型的计算密集型
 */
#define TILE_DIM 32

__global__ void matrix_mul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ==================== 设备属性查询 ====================

void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("========================================\n");
    printf("GPU 设备信息\n");
    printf("========================================\n");
    printf("型号: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM 数量: %d\n", prop.multiProcessorCount);
    printf("全局显存: %.2f GB\n", prop.totalGlobalMem / 1e9);

    // 估算理论带宽 (简化计算)
    // 获取内存时钟频率 (CUDA 13.0+ 需要使用 cudaDeviceGetAttribute)
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    float peak_bandwidth = prop.memoryBusWidth * 2.0 * mem_clock_khz * 2 / 8.0 / 1e6;
    printf("理论内存带宽: %.1f GB/s\n", peak_bandwidth);

    printf("\n");

    // 常见 GPU 的峰值算力 (需要从规格表查询)
    printf("常见 GPU 峰值算力参考:\n");
    printf("  A100 (SM 8.0): FP32 ~19.5 TFLOPS, FP16 ~312 TFLOPS\n");
    printf("  RTX 4090 (SM 9.0): FP32 ~82.6 TFLOPS, FP16 ~330 TFLOPS\n");
    printf("  V100 (SM 7.0): FP32 ~15.7 TFLOPS, FP16 ~125 TFLOPS\n");
    printf("\n");

    printf("Roofline 脊点计算:\n");
    printf("  脊点 AI = 峰值算力 / 峰值带宽\n");
    printf("  例: A100 FP32 = 19500 GFLOPS / 2039 GB/s ≈ 9.6 FLOP/byte\n");
    printf("========================================\n\n");
}

// ==================== 计时辅助函数 ====================

template<typename KernelFunc, typename... Args>
float time_kernel(KernelFunc kernel, dim3 grid, dim3 block, Args... args) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    kernel<<<grid, block>>>(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时 (100 次)
    int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(args...);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

// ==================== 性能分析结构体 ====================

typedef struct {
    const char* name;
    float time_ms;
    float gflops;
    float bandwidth_gbps;
    float ai;  // Arithmetic Intensity
    const char* bound_type;
} KernelPerf;

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    int n = 1 << 24;  // 16M 元素
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
    }

    // 打印设备信息
    print_device_info();

    printf("测试配置: %d 元素 (%.2f MB)\n\n", n, n * sizeof(float) / 1024.0 / 1024.0);

    int blockSize = 256;

    // ==================== 测试 1: 向量加法 ====================
    printf("========================================\n");
    printf("测试 1: 向量加法 (AI = 0.083)\n");
    printf("========================================\n");

    {
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        CUDA_CHECK(cudaMemset(d_a, 0, bytes));
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));

        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = time_kernel(vector_add, dim3(gridSize), dim3(blockSize),
                               d_a, d_b, d_c, n);

        // 性能计算
        float flops = 1.0 * n;  // 每个 1 次加法
        float bytes_accessed = 3.0 * bytes;  // 2 读 + 1 写
        float ai = flops / bytes_accessed;

        float gflops = flops / (ms * 1e-3) / 1e9;
        float bandwidth = bytes_accessed / (ms * 1e-3) / 1e9;

        printf("计算强度 (AI): %.4f FLOP/byte\n", ai);
        printf("执行时间: %.4f ms\n", ms);
        printf("性能: %.2f GFLOPS\n", gflops);
        printf("带宽: %.2f GB/s\n", bandwidth);
        printf("瓶颈类型: 访存密集型 (Memory-Bound)\n");
        printf("优化建议: 提高带宽利用率 (向量化、降低精度)\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    printf("\n");

    // ==================== 测试 2: 向量 FMA ====================
    printf("========================================\n");
    printf("测试 2: 向量 FMA (AI = 0.125)\n");
    printf("========================================\n");

    {
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_c, *d_d;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        CUDA_CHECK(cudaMalloc(&d_d, bytes));

        CUDA_CHECK(cudaMemset(d_a, 0, bytes));
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));

        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = time_kernel(vector_fma, dim3(gridSize), dim3(blockSize),
                               d_a, d_b, d_c, d_d, n);

        float flops = 2.0 * n;  // 乘法 + 加法
        float bytes_accessed = 4.0 * bytes;  // 3 读 + 1 写
        float ai = flops / bytes_accessed;

        float gflops = flops / (ms * 1e-3) / 1e9;
        float bandwidth = bytes_accessed / (ms * 1e-3) / 1e9;

        printf("计算强度 (AI): %.4f FLOP/byte\n", ai);
        printf("执行时间: %.4f ms\n", ms);
        printf("性能: %.2f GFLOPS\n", gflops);
        printf("带宽: %.2f GB/s\n", bandwidth);
        printf("瓶颈类型: 访存密集型\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_d));
    }

    printf("\n");

    // ==================== 测试 3: 中等计算强度 ====================
    printf("========================================\n");
    printf("测试 3: 中等计算强度 (AI = 0.67)\n");
    printf("========================================\n");

    {
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        CUDA_CHECK(cudaMemset(d_a, 0, bytes));
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));

        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = time_kernel(compute_medium, dim3(gridSize), dim3(blockSize),
                               d_a, d_b, d_c, n);

        float flops = 8.0 * n;
        float bytes_accessed = 3.0 * bytes;
        float ai = flops / bytes_accessed;

        float gflops = flops / (ms * 1e-3) / 1e9;
        float bandwidth = bytes_accessed / (ms * 1e-3) / 1e9;

        printf("计算强度 (AI): %.4f FLOP/byte\n", ai);
        printf("执行时间: %.4f ms\n", ms);
        printf("性能: %.2f GFLOPS\n", gflops);
        printf("带宽: %.2f GB/s\n", bandwidth);
        printf("瓶颈类型: 过渡区域 (取决于具体 GPU)\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    printf("\n");

    // ==================== 测试 4: 高计算强度 ====================
    printf("========================================\n");
    printf("测试 4: 高计算强度 (AI = 2.5)\n");
    printf("========================================\n");

    {
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        CUDA_CHECK(cudaMemset(d_a, 0, bytes));
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));

        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = time_kernel(compute_intensive, dim3(gridSize), dim3(blockSize),
                               d_a, d_b, d_c, n);

        float flops = 30.0 * n;
        float bytes_accessed = 3.0 * bytes;
        float ai = flops / bytes_accessed;

        float gflops = flops / (ms * 1e-3) / 1e9;
        float bandwidth = bytes_accessed / (ms * 1e-3) / 1e9;

        printf("计算强度 (AI): %.4f FLOP/byte\n", ai);
        printf("执行时间: %.4f ms\n", ms);
        printf("性能: %.2f GFLOPS\n", gflops);
        printf("带宽: %.2f GB/s\n", bandwidth);
        printf("瓶颈类型: 计算密集型 (Compute-Bound)\n");
        printf("优化建议: 提高计算效率 (并行、SIMD、Tensor Core)\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    printf("\n");

    // ==================== 测试 5: 优化技术对比 ====================
    printf("========================================\n");
    printf("测试 5: 优化技术对比\n");
    printf("========================================\n");

    printf("\n%-20s %12s %12s %12s %12s\n",
           "版本", "时间(ms)", "GFLOPS", "带宽(GB/s)", "AI");
    printf("-------------------------------------------------------------------------\n");

    // FP32 标量
    {
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = time_kernel(vector_add, dim3(gridSize), dim3(blockSize),
                               d_a, d_b, d_c, n);

        float bandwidth = 3.0 * bytes / (ms * 1e-3) / 1e9;
        float gflops = 1.0 * n / (ms * 1e-3) / 1e9;
        printf("%-20s %12.4f %12.1f %12.1f %12.4f\n",
               "FP32 标量", ms, gflops, bandwidth, 1.0f/12);
        printf("  -> 基准版本\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    // FP32 float4
    {
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        int n4 = n / 4;
        int gridSize = (n4 + blockSize - 1) / blockSize;
        float ms = time_kernel(vector_add_float4, dim3(gridSize), dim3(blockSize),
                               (float4*)d_a, (float4*)d_b, (float4*)d_c, n);

        float bandwidth = 3.0 * bytes / (ms * 1e-3) / 1e9;
        float gflops = 1.0 * n / (ms * 1e-3) / 1e9;
        printf("%-20s %12.4f %12.1f %12.1f %12.4f\n",
               "FP32 float4", ms, gflops, bandwidth, 1.0f/12);
        printf("  -> 向量化提高带宽利用率\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    // FP16 half2
    {
        size_t bytes = n * sizeof(half);
        half *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        int n2 = n / 2;
        int gridSize = (n2 + blockSize - 1) / blockSize;
        float ms = time_kernel(vector_add_half2, dim3(gridSize), dim3(blockSize),
                               (half2*)d_a, (half2*)d_b, (half2*)d_c, n);

        float bandwidth = 3.0 * bytes / (ms * 1e-3) / 1e9;
        float gflops = 1.0 * n / (ms * 1e-3) / 1e9;
        printf("%-20s %12.4f %12.1f %12.1f %12.4f\n",
               "FP16 half2", ms, gflops, bandwidth, 1.0f/6);
        printf("  -> FP16 使 AI 翻倍，向右移动!\n");

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    printf("-------------------------------------------------------------------------\n");

    // ==================== Roofline 总结 ====================
    printf("\n========================================\n");
    printf("Roofline 模型总结\n");
    printf("========================================\n");
    printf("\n");
    printf("Roofline 图示:\n\n");
    printf("  性能 (GFLOPS)\n");
    printf("      ^\n");
    printf("      │        计算密集区 (Compute-Bound)\n");
    printf("      │       ┌────────────────\n");
    printf("      │      ╱ ← 峰值算力天花板\n");
    printf("      │     ╱\n");
    printf("      │    ╱\n");
    printf("      │   ╱\n");
    printf("      │  ╱ ← 脊点 (Ridge Point)\n");
    printf("      │ ╱\n");
    printf("      │╱ 访存密集区 (Memory-Bound)\n");
    printf("      └───────────────────────────→ AI\n");
    printf("        0.1    1      10     100\n");
    printf("\n");
    printf("各算子在 Roofline 图上的位置:\n");
    printf("  vector_add:      AI ≈ 0.083 (极左，访存密集)\n");
    printf("  vector_fma:      AI ≈ 0.125 (访存密集)\n");
    printf("  compute_medium:  AI ≈ 0.67  (过渡区)\n");
    printf("  compute_intense: AI ≈ 2.5   (计算密集)\n");
    printf("  matrix_mul:      AI ≈ 682   (极右，计算密集)\n");
    printf("\n");
    printf("优化策略:\n");
    printf("  1. 脊点以左 (访存密集):\n");
    printf("     - 提高带宽利用率 (向量化访问)\n");
    printf("     - 降低精度 (FP16 -> AI 翻倍)\n");
    printf("     - 减少访存次数 (算子融合)\n");
    printf("\n");
    printf("  2. 脊点以右 (计算密集):\n");
    printf("     - 提高计算效率 (并行、SIMD)\n");
    printf("     - 使用 Tensor Core\n");
    printf("     - 优化指令级并行\n");
    printf("\n");
    printf("使用 Nsight Compute 查看详细 Roofline:\n");
    printf("  ncu --set full -o report ./roofline_demo\n");
    printf("  ncu-ui report.ncu-rep\n");
    printf("========================================\n");

    return 0;
}