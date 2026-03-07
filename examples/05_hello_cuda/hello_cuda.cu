/*
 * hello_cuda.cu - 第五章：第一个 CUDA 程序
 *
 * 这是 CUDA 入门的第一个示例程序，演示：
 * 1. CUDA 程序的基本结构
 * 2. 如何编写内核函数（kernel function）
 * 3. 如何从 CPU 调用 GPU 内核
 * 4. 如何获取线程索引信息
 *
 * 编译方法：
 *   使用 CMake: mkdir build && cd build && cmake .. && make
 *   或直接编译: nvcc -arch=sm_80 hello_cuda.cu -o hello_cuda
 *
 * 运行方法：
 *   ./hello_cuda
 *
 * 适合：CUDA 初学者，没有任何 CUDA 基础的开发者
 */

// 包含 CUDA 运行时 API 的头文件
// 这个头文件提供了所有 CUDA 的基本函数和类型
// 包括：cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize 等
#include <cuda_runtime.h>

// 包含标准输入输出头文件，用于 printf 函数
#include <stdio.h>

/*
 * __global__ 关键字说明：
 * ================
 *
 * __global__ 是 CUDA 的函数限定符，表示这是一个内核函数（kernel function）
 *
 * 特点：
 * 1. 在 GPU 上执行，由 CPU 调用
 * 2. 每个 __global__ 函数会被 GPU 上的多个线程同时执行（并行）
 * 3. 返回类型必须是 void
 * 4. 不支持可变参数
 * 5. 不支持静态变量
 * 6. 函数参数通过常量内存传递（有大小限制）
 *
 * 调用语法：
 * kernel_name<<<grid_size, block_size>>>(arguments...)
 *
 * - grid_size: 网格大小，指定启动多少个块（block）
 * - block_size: 块大小，指定每个块有多少个线程（thread）
 */
__global__ void hello_cuda()
{
    /*
     * CUDA 线程索引系统说明：
     * ====================
     *
     * blockIdx.x - 当前线程所在的块索引（block index）
     * - 在 x 维度上的块编号，从 0 开始
     * - 例如：如果启动了 2 个块，blockIdx.x 可能是 0 或 1
     *
     * threadIdx.x - 当前线程在块内的线程索引（thread index）
     * - 在 x 维度上的线程编号，从 0 开始
     * - 例如：如果每个块有 256 个线程，threadIdx.x 范围是 0-255
     *
     * blockDim.x - 每个块在 x 维度上的线程数量
     * - 这是一个常量，在内核启动时确定
     * - 所有块有相同的 blockDim
     *
     * gridDim.x - 网格在 x 维度上的块数量
     * - 表示总共有多少个块
     *
     * 全局线程索引计算公式：
     * global_idx = blockIdx.x * blockDim.x + threadIdx.x
     */

    // 打印 "Hello from GPU!" 以及当前线程的信息
    // 注意：printf 在 CUDA 内核中也可以使用（从 CUDA 4.0 开始支持）
    // 这是调试 CUDA 程序的常用方法
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

/*
 * 一个稍微复杂一点的内核函数
 * 演示如何计算全局线程索引
 */
__global__ void hello_with_global_id()
{
    // 计算全局线程索引
    // 这是 CUDA 编程中最常用的索引计算方式
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello! 全局ID: %2d (Block %d, Thread %d)\n",
           global_id, blockIdx.x, threadIdx.x);
}

/*
 * 主函数 - 在 CPU 上执行
 * 这是程序的入口点
 */
int main()
{
    printf("=========================================\n");
    printf("   第五章：第一个 CUDA 程序 - Hello CUDA\n");
    printf("=========================================\n\n");

    // ==================== 第一部分：最简单的内核 ====================
    printf("【第一部分】最简单的内核调用\n");
    printf("-----------------------------------------\n");
    printf("启动配置: <<<1, 1>>> (1 个块, 每块 1 个线程)\n");
    printf("总线程数: 1\n\n");

    /*
     * 内核启动语法说明：
     * kernel_name<<<grid_size, block_size>>>(arguments...)
     *
     * <<<1, 1>>> 表示：
     * - 第一个参数 1：启动 1 个块
     * - 第二个参数 1：每个块有 1 个线程
     * - 总线程数 = 1 * 1 = 1
     *
     * 这是 CUDA 中最简单的内核启动方式
     */
    hello_cuda<<<1, 1>>>();

    /*
     * cudaDeviceSynchronize() 说明：
     * =============================
     *
     * 内核启动是异步的，CPU 不会等待 GPU 执行完成
     * 这个函数会阻塞 CPU，直到 GPU 完成所有之前的任务
     *
     * 为什么需要同步？
     * 1. 确保 GPU 的 printf 输出显示在屏幕上
     * 2. 确保 GPU 完成计算后再访问结果
     * 3. 用于调试和性能测量
     *
     * 返回值：cudaSuccess 表示成功，其他值表示错误
     */
    cudaError_t err = cudaDeviceSynchronize();

    // 检查是否有错误发生
    if (err != cudaSuccess)
    {
        // cudaGetErrorString 将错误码转换为可读的错误信息
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("\n内核执行完成！\n\n");

    // ==================== 第二部分：多线程内核 ====================
    printf("【第二部分】多线程内核调用\n");
    printf("-----------------------------------------\n");
    printf("启动配置: <<<2, 4>>> (2 个块, 每块 4 个线程)\n");
    printf("总线程数: 2 * 4 = 8\n\n");

    // 启动 2 个块，每块 4 个线程，总共 8 个线程
    hello_with_global_id<<<2, 4>>>();

    // 等待 GPU 完成并检查错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("\n");

    // ==================== 总结 ====================
    printf("=========================================\n");
    printf("           学习要点总结\n");
    printf("=========================================\n\n");

    printf("1. CUDA 程序的基本结构:\n");
    printf("   - 包含 cuda_runtime.h 头文件\n");
    printf("   - 定义 __global__ 内核函数\n");
    printf("   - 在 main() 中调用内核\n");
    printf("   - 使用 cudaDeviceSynchronize() 同步\n\n");

    printf("2. 内核启动语法:\n");
    printf("   kernel<<<grid_size, block_size>>>(args)\n\n");

    printf("3. 线程索引变量:\n");
    printf("   - blockIdx.x: 块索引\n");
    printf("   - threadIdx.x: 线程索引\n");
    printf("   - blockDim.x: 块大小\n");
    printf("   - gridDim.x: 网格大小\n\n");

    printf("4. 全局索引计算:\n");
    printf("   global_id = blockIdx.x * blockDim.x + threadIdx.x\n\n");

    printf("=========================================\n");
    printf("   恭喜！你已经完成了第一个 CUDA 程序！\n");
    printf("=========================================\n");

    // 程序正常退出
    return 0;
}