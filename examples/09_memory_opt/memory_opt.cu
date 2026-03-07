/**
 * 第九章: CUDA 内存优化
 *
 * 学习目标:
 *   1. 理解合并访问 (Coalesced Access) 与非合并访问的区别
 *   2. 掌握 float4 向量化访存技术
 *   3. 学习减少内存事务的方法
 *   4. 对比不同访问模式的性能差异
 *
 * 编译方法:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
 *   make
 *
 * 运行:
 *   ./memory_opt
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

// ==================== 合并访问 vs 非合并访问 ====================

/**
 * 合并访问 (Coalesced Access) - 最佳实践
 *
 * 特点:
 *   - 连续的线程访问连续的内存地址
 *   - 相邻线程 (threadIdx.x 相差1) 访问相邻地址
 *   - GPU 可以将多个访问合并为一个内存事务
 *
 * 内存访问模式:
 *   Thread 0: addr[0]
 *   Thread 1: addr[1]
 *   Thread 2: addr[2]
 *   ...
 */
__global__ void coalesced_access(float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 连续线程访问连续地址 - 合并!
        out[idx] = data[idx] * 2.0f;
    }
}

/**
 * 非合并访问 (Strided Access) - 性能较差
 *
 * 特点:
 *   - 连续线程访问跨度较大的地址
 *   - 每次访问生成单独的内存事务
 *   - 带宽利用率低
 *
 * 内存访问模式:
 *   Thread 0: addr[0]
 *   Thread 1: addr[stride]     <- 跨度大
 *   Thread 2: addr[2*stride]
 *   ...
 */
__global__ void strided_access(float* data, float* out, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;  // 步长访问

    if (idx < n) {
        out[idx] = data[idx] * 2.0f;
    }
}

/**
 * 非合并访问 (Offset Access) - 性能较差
 *
 * 特点:
 *   - 访问虽然连续，但偏移导致未对齐
 *   - 一个 warp 的访问可能跨越多个缓存行
 */
__global__ void offset_access(float* data, float* out, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < n && idx >= offset) {
        out[idx - offset] = data[idx] * 2.0f;
    }
}

// ==================== 向量化访存 ====================

/**
 * 标量访问 - 基准版本
 *
 * 每个线程:
 *   - 读取 1 个 float (4 字节)
 *   - 执行 1 次内存加载事务
 */
__global__ void scalar_access(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * float2 向量化访问
 *
 * 每个线程:
 *   - 读取 1 个 float2 (8 字节)
 *   - 处理 2 个 float 元素
 *   - 减少内存事务次数
 */
__global__ void vectorized_float2(float2* a, float2* b, float2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        // 一次加载 8 字节
        float2 va = a[idx];
        float2 vb = b[idx];

        float2 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;

        c[idx] = vc;
    }
}

/**
 * float4 向量化访问 - 最佳实践
 *
 * 每个线程:
 *   - 读取 1 个 float4 (16 字节)
 *   - 处理 4 个 float 元素
 *   - 128 位对齐，最大化带宽利用
 *
 * 优势:
 *   - 减少 GPU 指令数
 *   - 提高内存总线利用率
 *   - 适合 SIMD 操作
 */
__global__ void vectorized_float4(float4* a, float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        // 一次加载 16 字节 - 128 位对齐
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

// ==================== 共享内存优化示例 ====================

/**
 * 使用共享内存的矩阵转置 - 演示 bank conflict 优化
 *
 * 全局内存访问模式:
 *   - 读取: 合并访问 (行优先)
 *   - 写入: 非合并访问 (列优先)
 *
 * 使用共享内存:
 *   - 减少全局内存访问
 *   - 可以优化 bank conflict
 */
#define TILE_SIZE 32

__global__ void transpose_naive(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 列访问 - 非合并!
        output[x * height + y] = input[y * width + x];
    }
}

__global__ void transpose_shared(float* input, float* output, int width, int height) {
    // 声明共享内存
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 从全局内存加载到共享内存 (合并读取)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // 计算转置后的位置
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // 从共享内存写入全局内存 (合并写入)
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ==================== 计时辅助函数 ====================

template<typename KernelFunc, typename... Args>
float benchmark_kernel(KernelFunc kernel,
                       dim3 grid, dim3 block,
                       int iterations,
                       Args... args) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    kernel<<<grid, block>>>(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(args...);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    int n = 1 << 24;  // 16M 元素
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
    }

    size_t bytes = n * sizeof(float);

    printf("========================================\n");
    printf("CUDA 内存优化演示\n");
    printf("========================================\n");
    printf("元素数量: %d (%.2f MB)\n", n, bytes / 1024.0 / 1024.0);
    printf("线程块大小: 256\n\n");

    // 分配内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int iterations = 100;

    // ==================== 测试 1: 合并访问 vs 非合并访问 ====================
    printf("========================================\n");
    printf("测试 1: 内存访问模式对比\n");
    printf("========================================\n");

    printf("\n%-20s %12s %12s %12s\n", "访问模式", "时间(ms)", "带宽(GB/s)", "相对性能");
    printf("------------------------------------------------------------\n");

    // 合并访问
    {
        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = benchmark_kernel(coalesced_access,
                                    dim3(gridSize), dim3(blockSize),
                                    iterations,
                                    d_a, d_c, n);

        float bandwidth = (2.0 * bytes / ms) / 1e6;  // 1读+1写
        printf("%-20s %12.4f %12.1f %12s\n",
               "合并访问", ms, bandwidth, "100%");
    }

    // 跨度访问 (stride=2)
    {
        int stride = 2;
        int gridSize = (n / stride + blockSize - 1) / blockSize;
        float ms = benchmark_kernel(strided_access,
                                    dim3(gridSize), dim3(blockSize),
                                    iterations,
                                    d_a, d_c, n, stride);

        float bandwidth = (2.0 * bytes / stride / ms) / 1e6;
        printf("%-20s %12.4f %12.1f %12.0f%%\n",
               "跨度访问(s=2)", ms, bandwidth,
               100.0 * bandwidth / (2.0 * bytes / 0.001 / 1e6) * 0.001);
    }

    // 跨度访问 (stride=32)
    {
        int stride = 32;
        int gridSize = (n / stride + blockSize - 1) / blockSize;
        float ms = benchmark_kernel(strided_access,
                                    dim3(gridSize), dim3(blockSize),
                                    iterations,
                                    d_a, d_c, n, stride);

        float bandwidth = (2.0 * bytes / stride / ms) / 1e6;
        printf("%-20s %12.4f %12.1f %12.0f%%\n",
               "跨度访问(s=32)", ms, bandwidth,
               100.0 * bandwidth / (2.0 * bytes / 0.001 / 1e6) * 0.001);
    }

    printf("------------------------------------------------------------\n");
    printf("说明: 跨度越大，缓存命中率越低，性能越差\n");

    // ==================== 测试 2: 向量化访问 ====================
    printf("\n========================================\n");
    printf("测试 2: 向量化访存对比\n");
    printf("========================================\n");

    printf("\n%-20s %12s %12s %12s\n", "访存类型", "时间(ms)", "带宽(GB/s)", "加速比");
    printf("------------------------------------------------------------\n");

    float baseline_ms;

    // 标量访问
    {
        int gridSize = (n + blockSize - 1) / blockSize;
        float ms = benchmark_kernel(scalar_access,
                                    dim3(gridSize), dim3(blockSize),
                                    iterations,
                                    d_a, d_b, d_c, n);

        baseline_ms = ms;
        float bandwidth = (3.0 * bytes / ms) / 1e6;  // 2读+1写
        printf("%-20s %12.4f %12.1f %12s\n",
               "float (标量)", ms, bandwidth, "1.00x");
    }

    // float2 向量化
    {
        int n2 = n / 2;
        int gridSize = (n2 + blockSize - 1) / blockSize;
        float ms = benchmark_kernel(vectorized_float2,
                                    dim3(gridSize), dim3(blockSize),
                                    iterations,
                                    (float2*)d_a, (float2*)d_b, (float2*)d_c, n);

        float bandwidth = (3.0 * bytes / ms) / 1e6;
        printf("%-20s %12.4f %12.1f %12.2fx\n",
               "float2 (向量)", ms, bandwidth, baseline_ms / ms);
    }

    // float4 向量化
    {
        int n4 = n / 4;
        int gridSize = (n4 + blockSize - 1) / blockSize;
        float ms = benchmark_kernel(vectorized_float4,
                                    dim3(gridSize), dim3(blockSize),
                                    iterations,
                                    (float4*)d_a, (float4*)d_b, (float4*)d_c, n);

        float bandwidth = (3.0 * bytes / ms) / 1e6;
        printf("%-20s %12.4f %12.1f %12.2fx\n",
               "float4 (向量)", ms, bandwidth, baseline_ms / ms);
    }

    printf("------------------------------------------------------------\n");
    printf("说明: float4 向量化可显著提升带宽利用率\n");

    // ==================== 测试 3: 矩阵转置 ====================
    printf("\n========================================\n");
    printf("测试 3: 矩阵转置优化\n");
    printf("========================================\n");

    int width = 4096;
    int height = 4096;
    size_t mat_bytes = width * height * sizeof(float);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, mat_bytes));
    CUDA_CHECK(cudaMemset(d_input, 0, mat_bytes));

    printf("\n%-25s %12s %12s\n", "转置方法", "时间(ms)", "带宽(GB/s)");
    printf("------------------------------------------------\n");

    // 朴素转置
    {
        dim3 block(32, 32);
        dim3 grid((width + 31) / 32, (height + 31) / 32);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 预热
        transpose_naive<<<grid, block>>>(d_input, d_output, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            transpose_naive<<<grid, block>>>(d_input, d_output, width, height);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 100;

        float bandwidth = (2.0 * mat_bytes / ms) / 1e6;
        printf("%-25s %12.4f %12.1f\n", "朴素转置 (非合并写)", ms, bandwidth);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // 共享内存转置
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
                  (height + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 预热
        transpose_shared<<<grid, block>>>(d_input, d_output, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            transpose_shared<<<grid, block>>>(d_input, d_output, width, height);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 100;

        float bandwidth = (2.0 * mat_bytes / ms) / 1e6;
        printf("%-25s %12.4f %12.1f\n", "共享内存转置", ms, bandwidth);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("------------------------------------------------\n");
    printf("说明: 共享内存优化了全局内存访问模式\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // ==================== 清理 ====================
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    // ==================== 总结 ====================
    printf("\n========================================\n");
    printf("内存优化要点总结\n");
    printf("========================================\n");
    printf("\n1. 合并访问 (Coalesced Access):\n");
    printf("   - 相邻线程访问相邻内存地址\n");
    printf("   - Warp 内访问合并为最少的内存事务\n");
    printf("   - 避免跨度访问和未对齐访问\n");
    printf("\n2. 向量化访存:\n");
    printf("   - 使用 float2/float4 一次读取多个数据\n");
    printf("   - 减少内存事务次数\n");
    printf("   - 128 位对齐获得最佳性能\n");
    printf("\n3. 共享内存:\n");
    printf("   - 用于优化全局内存访问模式\n");
    printf("   - 注意避免 bank conflict\n");
    printf("   - 合理规划数据布局\n");
    printf("========================================\n");

    return 0;
}