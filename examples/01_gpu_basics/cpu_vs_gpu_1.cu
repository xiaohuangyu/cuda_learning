/*
 * CPU vs GPU 矩阵乘法对比示例
 * ==========================
 * 本程序演示相同的矩阵乘法任务在 CPU 和 GPU 上的执行时间对比：
 *   C = A * B
 *
 * CPU：朴素三重循环（row-major）
 * GPU：tiled kernel + shared memory（row-major）
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// CUDA 运行时头文件
#include <cuda_runtime.h>

// 矩阵维度：A(dim x dim) * B(dim x dim) = C(dim x dim)
// 选择一个适中大小，保证演示时 CPU 不会太慢
#define DIM 1024

// GPU 计算 tile 大小（必须和 shared memory 数组维度一致）
#define TILE_SIZE 16

// 结果校验的容差（浮点误差可能来自累加顺序/FMA）
#define EPS 1e-2f

// CUDA 调用错误检查（便于定位问题）
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                              \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// ============================================================================
// CPU 版本的矩阵乘法（朴素三重循环）
// ============================================================================
void matmul_cpu(const float *A, const float *B, float *C, int dim) {
    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++) {
                sum += A[row * dim + k] * B[k * dim + col];
            }
            C[row * dim + col] = sum;
        }
    }
}

// ============================================================================
// GPU 版本的矩阵乘法（tiled + shared memory）
// ============================================================================
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C, int dim) {
    // 共享内存：分别缓存 A 和 B 的一个 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 以 TILE_SIZE 为步长遍历 K 维
    for (int t = 0; t < dim; t += TILE_SIZE) {
        // 加载 A 的 tile 到 shared memory
        int aCol = t + threadIdx.x;
        if (row < dim && aCol < dim) {
            As[threadIdx.y][threadIdx.x] = A[row * dim + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的 tile 到 shared memory
        int bRow = t + threadIdx.y;
        if (bRow < dim && col < dim) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * dim + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算一个 tile 内的部分积
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < dim && col < dim) {
        C[row * dim + col] = sum;
    }
}

// ============================================================================
// 辅助函数：获取当前时间（毫秒）
// ============================================================================
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("========================================\n");
    printf("    CPU vs GPU 矩阵乘法性能对比演示程序 _1\n");
    printf("========================================\n\n");

    const int dim = DIM;
    const size_t bytes = (size_t)dim * (size_t)dim * sizeof(float);

    // ------------------------------------------------------------------------
    // 1. 在主机（CPU）内存中分配矩阵空间
    // ------------------------------------------------------------------------
    printf("[步骤1] 分配主机矩阵内存...\n");
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_cpu = (float*)malloc(bytes);
    float *h_c_gpu = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu) {
        printf("错误：主机内存分配失败！\n");
        return -1;
    }
    printf("    矩阵维度: %d x %d\n", dim, dim);
    printf("    每个矩阵大小: %.2f MB\n\n", (double)bytes / 1024.0 / 1024.0);

    // ------------------------------------------------------------------------
    // 2. 初始化数据
    // ------------------------------------------------------------------------
    printf("[步骤2] 初始化输入数据...\n");
    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            // 让数据范围适中，避免数值太大导致误差判断困难
            h_a[row * dim + col] = (float)((row * dim + col) % 100) / 10.0f;
            h_b[row * dim + col] = (float)((row + col) % 100) / 10.0f;
        }
    }
    printf("    数据初始化完成\n\n");

    // ========================================================================
    // CPU 执行部分：朴素矩阵乘法
    // ========================================================================
    printf("[步骤3] 在CPU上执行矩阵乘法(朴素)...\n");
    double cpu_start = get_time_ms();

    matmul_cpu(h_a, h_b, h_c_cpu, dim);

    double cpu_end = get_time_ms();
    double cpu_time = cpu_end - cpu_start;
    printf("    CPU 执行时间: %.3f 毫秒\n\n", cpu_time);

    // ========================================================================
    // GPU 执行部分：tiled + shared memory 矩阵乘法
    // ========================================================================
    printf("[步骤4] 在GPU上执行矩阵乘法(tiled)...\n");

    // 4.4 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 4.5 执行GPU核函数（事件计时）
    CUDA_CHECK(cudaEventRecord(start));

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 4.2 将数据从主机拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((dim + TILE_SIZE - 1) / TILE_SIZE, (dim + TILE_SIZE - 1) / TILE_SIZE);
    printf("    Grid: (%d, %d), Block: (%d, %d)\n", (int)grid.x, (int)grid.y, (int)threads.x, (int)threads.y);   

    matmul_tiled_kernel<<<grid, threads>>>(d_a, d_b, d_c, dim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 4.6 计算GPU执行时间
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("    GPU 执行时间: %.3f 毫秒\n\n", gpu_time);

    // 4.7 将结果从设备拷贝回主机
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // ========================================================================
    // 验证结果
    // ========================================================================
    printf("[步骤5] 验证计算结果...\n");
    int correct = 1;
    float max_err = 0.0f;
    int mismatch_idx = -1;
    for (int idx = 0; idx < dim * dim; idx++) {
        float cpu_v = h_c_cpu[idx];
        float gpu_v = h_c_gpu[idx];
        float err = fabsf(cpu_v - gpu_v);
        if (err > max_err) {
            max_err = err;
        }
        if (err > EPS) {
            correct = 0;
            mismatch_idx = idx;
            break;
        }
    }
    if (correct) printf("    验证通过！CPU 和 GPU 结果一致 (max_err=%.6f)\n\n", max_err);
    else {
        int row = mismatch_idx / dim;
        int col = mismatch_idx % dim;
        printf("    验证失败！在 C[%d,%d] (idx=%d) 处误差过大\n", row, col, mismatch_idx);
        printf("    CPU: %.6f, GPU: %.6f, abs_err=%.6f (EPS=%.6f)\n\n",
               h_c_cpu[mismatch_idx], h_c_gpu[mismatch_idx], fabsf(h_c_cpu[mismatch_idx] - h_c_gpu[mismatch_idx]), EPS);
    }

    // ========================================================================
    // 性能对比总结
    // ========================================================================
    printf("========================================\n");
    printf("              性能对比结果\n");
    printf("========================================\n");
    printf("    矩阵维度: %d x %d\n", dim, dim);
    printf("    CPU 时间: %.3f 毫秒\n", cpu_time);
    printf("    GPU 时间: %.3f 毫秒\n", gpu_time);
    printf("    加速比: %.2fx\n", cpu_time / gpu_time);
    printf("========================================\n\n");

    printf("性能分析说明：\n");
    printf("1. GPU 通过大量并行线程处理矩阵元素\n");
    printf("2. tiled kernel 使用 shared memory 减少全局访存\n");
    printf("3. 数据传输开销也会影响总体性能（本例计时的是 kernel 时间）\n");
    printf("4. 此示例用于展示 CPU/GPU 对比与计时校验流程\n");

    // ------------------------------------------------------------------------
    // 清理资源
    // ------------------------------------------------------------------------
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}
