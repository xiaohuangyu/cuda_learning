/*
 * CPU vs GPU 对比示例
 * ================
 * 本程序演示相同的计算任务在CPU和GPU上的执行时间对比
 *
 * 任务：对大型数组进行简单的向量加法运算
 * C[i] = A[i] + B[i]
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA 运行时头文件
#include <cuda_runtime.h>

// 数据大小（1000万个元素）
#define N 10000000

// ============================================================================
// CPU 版本的向量加法
// ============================================================================
/**
 * 在CPU上执行向量加法
 * @param a 输入向量A
 * @param b 输入向量B
 * @param c 输出向量C
 * @param n 向量长度
 */
void vector_add_cpu(float *a, float *b, float *c, int n) {
    // CPU使用循环逐个处理每个元素
    // 这是典型的串行执行方式
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// GPU 版本的向量加法（核函数）
// ============================================================================
/**
 * 在GPU上执行向量加法的核函数
 * 使用__global__修饰符表示这是一个可以从CPU调用的GPU函数
 *
 * 核函数的特点：
 * 1. 由CPU调用，在GPU上执行
 * 2. 被多个线程并行执行
 * 3. 每个线程处理一个或多个数据元素
 */
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    // 计算当前线程应该处理的元素索引
    // blockIdx.x: 当前线程块在网格中的索引
    // blockDim.x: 线程块中线程的数量
    // threadIdx.x: 当前线程在线程块中的索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保不越界访问
    // 当线程数量大于数据量时，部分线程会跳过计算
    if (i < n) {
        c[i] = a[i] + b[i];
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
    printf("    CPU vs GPU 性能对比演示程序 _1\n");
    printf("========================================\n\n");

    // ------------------------------------------------------------------------
    // 1. 在主机（CPU）内存中分配空间
    // ------------------------------------------------------------------------
    printf("[步骤1] 分配主机内存...\n");
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c_cpu = (float*)malloc(N * sizeof(float));
    float *h_c_gpu = (float*)malloc(N * sizeof(float));

    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu) {
        printf("错误：主机内存分配失败！\n");
        return -1;
    }
    printf("    已分配 %d 个浮点数 (%.2f MB)\n\n", N, N * sizeof(float) * 4 / 1024.0 / 1024.0);

    // ------------------------------------------------------------------------
    // 2. 初始化数据
    // ------------------------------------------------------------------------
    printf("[步骤2] 初始化输入数据...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    printf("    数据初始化完成\n\n");

    // ========================================================================
    // CPU 执行部分
    // ========================================================================
    printf("[步骤3] 在CPU上执行向量加法...\n");
    double cpu_start = get_time_ms();

    // 调用CPU版本的向量加法
    vector_add_cpu(h_a, h_b, h_c_cpu, N);

    double cpu_end = get_time_ms();
    double cpu_time = cpu_end - cpu_start;
    printf("    CPU 执行时间: %.3f 毫秒\n\n", cpu_time);

    // ========================================================================
    // GPU 执行部分
    // ========================================================================
    printf("[步骤4] 在GPU上执行向量加法...\n");

    // 4.1 在设备（GPU）内存中分配空间
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 4.2 将数据从主机拷贝到设备
    // cudaMemcpyHostToDevice: 从CPU拷贝到GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4.3 配置线程网格和线程块
    // 每个线程块包含256个线程
    int threads_per_block = 256;

    // 计算需要的线程块数量（向上取整）
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    printf("    线程块数量: %d\n", blocks_per_grid);
    printf("    每块线程数: %d\n", threads_per_block);
    printf("    总线程数: %d\n", blocks_per_grid * threads_per_block);

    // 4.4 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 4.5 执行GPU核函数
    cudaEventRecord(start);  // 记录开始时间

    // 核函数调用语法：kernel<<<网格大小, 线程块大小>>>(参数...)
    vector_add_gpu<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);   // 记录结束时间
    cudaEventSynchronize(stop);  // 等待GPU执行完成

    // 4.6 计算GPU执行时间
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("    GPU 执行时间: %.3f 毫秒\n\n", gpu_time);

    // 4.7 将结果从设备拷贝回主机
    // cudaMemcpyDeviceToHost: 从GPU拷贝到CPU
    cudaMemcpy(h_c_gpu, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ========================================================================
    // 验证结果
    // ========================================================================
    printf("[步骤5] 验证计算结果...\n");
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            printf("    错误：索引 %d 处结果不匹配 (CPU: %.0f, GPU: %.0f)\n",
                   i, h_c_cpu[i], h_c_gpu[i]);
            break;
        }
    }
    if (correct) {
        printf("    验证通过！CPU和GPU结果一致\n\n");
    }

    // ========================================================================
    // 性能对比总结
    // ========================================================================
    printf("========================================\n");
    printf("              性能对比结果\n");
    printf("========================================\n");
    printf("    数据量: %d 个浮点数\n", N);
    printf("    CPU 时间: %.3f 毫秒\n", cpu_time);
    printf("    GPU 时间: %.3f 毫秒\n", gpu_time);
    printf("    加速比: %.2fx\n", cpu_time / gpu_time);
    printf("========================================\n\n");

    // 解释加速原因
    printf("性能分析说明：\n");
    printf("1. GPU通过大规模并行（数千个线程同时执行）获得加速\n");
    printf("2. 对于简单运算，数据传输开销可能抵消并行收益\n");
    printf("3. 数据量越大，GPU优势越明显\n");
    printf("4. 此示例展示了GPU计算的基本模式\n");

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
