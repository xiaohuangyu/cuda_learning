/*
 * CUDA 基础语法示例
 * =================
 * 本程序演示CUDA编程中的三种函数类型：
 * 1. __global__ - 核函数，在GPU上执行，由CPU调用
 * 2. __device__ - 设备函数，在GPU上执行，只能被GPU函数调用
 * 3. __host__   - 主机函数，在CPU上执行（默认函数类型）
 */

#include <stdio.h>

// ============================================================================
// __device__ 函数示例
// ============================================================================
/**
 * 设备函数：只能在GPU上调用的函数
 * 用于封装在核函数中重复使用的代码
 *
 * 特点：
 * 1. 在GPU上执行
 * 2. 只能被__global__或__device__函数调用
 * 3. 不能直接从CPU调用
 *
 * @param x 输入值
 * @return 计算结果
 */
__device__ float device_square(float x) {
    // 这个函数在GPU上执行
    // 可以被其他GPU函数调用
    return x * x;
}

/**
 * 另一个设备函数示例：计算两个数的和
 * 展示设备函数可以有多个参数
 */
__device__ float device_add(float a, float b) {
    return a + b;
}

/**
 * 设备函数可以调用其他设备函数
 * 展示GPU函数的模块化编程
 */
__device__ float device_compute(float a, float b) {
    // 调用其他设备函数
    float sum = device_add(a, b);
    float result = device_square(sum);
    return result;
}

// ============================================================================
// __global__ 函数示例（核函数）
// ============================================================================
/**
 * 核函数：GPU程序的入口点
 * 由CPU调用，在GPU上并行执行
 *
 * 特点：
 * 1. 必须返回void类型
 * 2. 由CPU调用，在GPU上执行
 * 3. 被多个线程并行执行
 * 4. 调用时使用特殊的 <<<>>> 语法
 *
 * 本示例演示：
 * - 如何调用__device__函数
 * - 线程索引的计算
 */
__global__ void kernel_basic(float *input, float *output, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < n) {
        // 在核函数中调用设备函数
        // 注意：这里不能调用__host__函数！
        output[idx] = device_square(input[idx]);
    }
}

/**
 * 展示更复杂的核函数
 * 调用多个设备函数完成复合计算
 */
__global__ void kernel_complex(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 调用复合设备函数
        result[idx] = device_compute(a[idx], b[idx]);
    }
}

/**
 * 展示核函数中的打印功能
 * 注意：printf在核函数中也可使用，但输出是缓冲的
 */
__global__ void kernel_print_info() {
    // 每个线程都会执行这段代码
    // blockIdx 和 threadIdx 是内置变量
    printf("线程信息: blockIdx=(%d,%d,%d), threadIdx=(%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

// ============================================================================
// __host__ 函数示例
// ============================================================================
/**
 * 主机函数：在CPU上执行的普通函数
 * 这是C/C++函数的默认类型
 *
 * 特点：
 * 1. 在CPU上执行
 * 2. 不能调用__device__函数
 * 3. 可以调用__global__函数（启动核函数）
 */
void host_function_example() {
    printf("这是一个主机（CPU）函数\n");
    printf("主机函数不能调用设备函数\n");
    printf("主机函数可以启动核函数\n");
}

/**
 * 辅助函数：验证计算结果
 */
void verify_result(float *expected, float *actual, int n, const char *test_name) {
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (expected[i] != actual[i]) {
            passed = false;
            printf("  [%s] 失败：索引 %d 处不匹配\n", test_name, i);
            break;
        }
    }
    if (passed) {
        printf("  [%s] 通过！\n", test_name);
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("========================================\n");
    printf("    CUDA 函数类型演示程序\n");
    printf("========================================\n\n");

    // ------------------------------------------------------------------------
    // 第一部分：函数类型说明
    // ------------------------------------------------------------------------
    printf("【CUDA 函数类型说明】\n\n");

    printf("1. __global__ 核函数:\n");
    printf("   - 在GPU上执行\n");
    printf("   - 由CPU调用\n");
    printf("   - 使用 <<<>>> 语法启动\n");
    printf("   - 必须返回 void\n\n");

    printf("2. __device__ 设备函数:\n");
    printf("   - 在GPU上执行\n");
    printf("   - 只能被GPU函数调用\n");
    printf("   - 不能被CPU直接调用\n\n");

    printf("3. __host__ 主机函数:\n");
    printf("   - 在CPU上执行（默认类型）\n");
    printf("   - 可以调用核函数\n");
    printf("   - 不能调用设备函数\n\n");

    printf("----------------------------------------\n");
    printf("【演示：主机函数调用】\n\n");
    host_function_example();

    // ------------------------------------------------------------------------
    // 第二部分：演示 __device__ 函数调用
    // ------------------------------------------------------------------------
    printf("----------------------------------------\n");
    printf("【演示：设备函数调用】\n\n");

    const int N = 8;
    float h_input[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_output[N];
    float h_a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_b[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_result[N];

    // 分配GPU内存
    float *d_input, *d_output, *d_a, *d_b, *d_result;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    // 拷贝数据到GPU
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 调用核函数（核函数内部会调用设备函数）
    printf("调用 kernel_basic（计算平方）:\n");
    kernel_basic<<<1, N>>>(d_input, d_output, N);

    // 获取结果
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("  输入: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_input[i]);
    printf("\n  输出: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_output[i]);
    printf("\n\n");

    // 验证结果
    float expected_output[N];
    for (int i = 0; i < N; i++) expected_output[i] = h_input[i] * h_input[i];
    verify_result(expected_output, h_output, N, "平方计算");

    // ------------------------------------------------------------------------
    // 第三部分：演示复合设备函数调用
    // ------------------------------------------------------------------------
    printf("\n----------------------------------------\n");
    printf("【演示：复合设备函数】\n\n");

    printf("计算 (a + b)^2:\n");
    kernel_complex<<<1, N>>>(d_a, d_b, d_result, N);
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("  a: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_a[i]);
    printf("\n  b: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_b[i]);
    printf("\n  (a+b)^2: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_result[i]);
    printf("\n\n");

    // 验证
    float expected_result[N];
    for (int i = 0; i < N; i++) {
        expected_result[i] = (h_a[i] + h_b[i]) * (h_a[i] + h_b[i]);
    }
    verify_result(expected_result, h_result, N, "复合计算");

    // ------------------------------------------------------------------------
    // 第四部分：演示线程索引
    // ------------------------------------------------------------------------
    printf("\n----------------------------------------\n");
    printf("【演示：线程索引信息】\n\n");

    printf("启动 2 个线程块，每块 4 个线程:\n\n");
    kernel_print_info<<<3, 2>>>();
    cudaDeviceSynchronize();  // 等待GPU完成

    // ------------------------------------------------------------------------
    // 第五部分：Host/Device 概念总结
    // ------------------------------------------------------------------------
    printf("\n========================================\n");
    printf("        Host/Device 概念总结\n");
    printf("========================================\n\n");

    printf("Host（主机）:\n");
    printf("  - 指 CPU 及其内存\n");
    printf("  - 执行 __host__ 函数\n");
    printf("  - 负责启动核函数\n");
    printf("  - 处理程序逻辑和 I/O\n\n");

    printf("Device（设备）:\n");
    printf("  - 指 GPU 及其内存\n");
    printf("  - 执行 __global__ 和 __device__ 函数\n");
    printf("  - 进行大规模并行计算\n\n");

    printf("数据流:\n");
    printf("  1. CPU 分配 GPU 内存 (cudaMalloc)\n");
    printf("  2. CPU 拷贝数据到 GPU (cudaMemcpy H2D)\n");
    printf("  3. CPU 启动核函数 (kernel<<<>>>)\n");
    printf("  4. GPU 并行执行计算\n");
    printf("  5. CPU 拷贝结果回 CPU (cudaMemcpy D2H)\n");
    printf("  6. CPU 释放 GPU 内存 (cudaFree)\n");

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
