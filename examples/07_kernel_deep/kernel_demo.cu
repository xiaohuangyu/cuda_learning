/*
 * kernel_demo.cu - 第七章：CUDA 内核深入理解
 *
 * 本示例深入演示 CUDA 函数限定符和内核启动配置：
 * 1. __global__ - 内核函数，在 GPU 上执行，由 CPU 调用
 * 2. __device__ - 设备函数，在 GPU 上执行，只能被 GPU 函数调用
 * 3. __host__ - 主机函数，在 CPU 上执行（默认）
 * 4. 不同的内核启动配置 <<<grid, block>>>
 * 5. 二维和三维的网格与块配置
 *
 * 编译方法：
 *   使用 CMake: mkdir build && cd build && cmake .. && make
 *   或直接编译: nvcc -arch=sm_80 kernel_demo.cu -o kernel_demo
 *
 * 适合：想要深入理解 CUDA 内核机制的开发者
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ==================== 函数限定符说明 ====================

/*
 * CUDA 函数限定符总结：
 * =====================
 *
 * ┌─────────────┬───────────────┬───────────────┬───────────────┐
 * │   限定符    │   执行位置    │   调用位置    │     备注      │
 * ├─────────────┼───────────────┼───────────────┼───────────────┤
 * │ __global__  │     GPU       │     CPU       │ 内核函数       │
 * │ __device__  │     GPU       │     GPU       │ 设备函数       │
 * │ __host__    │     CPU       │     CPU       │ 主机函数(默认) │
 * │ __device__  │     GPU       │   CPU/GPU     │ 可在两边调用   │
 * │ __host__    │               │               │               │
 * └─────────────┴───────────────┴───────────────┴───────────────┘
 */

// ==================== __device__ 函数示例 ====================

/*
 * __device__ 函数说明：
 * ====================
 *
 * - 在 GPU 上执行
 * - 只能被 __global__ 或其他 __device__ 函数调用
 * - 不能直接从 CPU 调用
 * - 用于封装 GPU 端的通用逻辑
 * - 内联函数，调用开销很小
 */
__device__ int compute_global_id(int block_id, int block_size, int thread_id)
{
    /*
     * 计算全局线程索引
     * 这是一个在 GPU 上执行的辅助函数
     * 被内核函数调用
     */
    return block_id * block_size + thread_id;
}

/*
 * 另一个 __device__ 函数示例
 * 执行实际的计算工作
 */
__device__ float add_values(float a, float b)
{
    return a + b;
}

/*
 * 更复杂的 __device__ 函数
 * 可以包含条件判断、循环等
 */
__device__ int factorial(int n)
{
    if (n <= 1) return 1;

    int result = 1;
    for (int i = 2; i <= n; i++)
    {
        result *= i;
    }
    return result;
}

// ==================== __global__ 内核函数示例 ====================

/*
 * __global__ 函数说明：
 * ====================
 *
 * - 在 GPU 上执行
 * - 从 CPU 调用（使用 <<<>>> 语法）
 * - 返回类型必须是 void
 * - 每个线程独立执行
 */

/*
 * 演示 __device__ 函数的调用
 * 在内核中调用设备函数
 */
__global__ void demo_device_function()
{
    // 调用 __device__ 函数计算全局 ID
    int global_id = compute_global_id(blockIdx.x, blockDim.x, threadIdx.x);

    printf("[demo_device_function] 线程 %d: blockIdx=%d, blockDim=%d, threadIdx=%d\n",
           global_id, blockIdx.x, blockDim.x, threadIdx.x);
}

/*
 * 演示多个 __device__ 函数的协作
 */
__global__ void demo_device_chain(float *a, float *b, float *c, int n)
{
    int idx = compute_global_id(blockIdx.x, blockDim.x, threadIdx.x);

    if (idx < n)
    {
        // 调用 __device__ 函数进行计算
        c[idx] = add_values(a[idx], b[idx]);
    }
}

/*
 * 演示 __device__ 函数中的复杂逻辑
 */
__global__ void demo_device_complex()
{
    int idx = threadIdx.x;
    int result = factorial(idx % 6);  // 0! 到 5!

    printf("线程 %d: %d! = %d\n", idx, idx % 6, result);
}

// ==================== 不同的启动配置示例 ====================

/*
 * 演示一维网格配置
 */
__global__ void demo_1d_config(int *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        output[idx] = idx;
        printf("[1D] 全局ID=%d, 块=%d, 块内线程=%d\n",
               idx, blockIdx.x, threadIdx.x);
    }
}

/*
 * 演示二维网格配置
 * dim3 grid(x, y) 和 dim3 block(x, y)
 */
__global__ void demo_2d_config()
{
    /*
     * 二维索引计算：
     *
     * 每个线程有二维坐标 (x, y)
     * - threadIdx.x, threadIdx.y: 块内坐标
     * - blockIdx.x, blockIdx.y: 块坐标
     */

    // 计算全局坐标
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算一维索引（假设行优先）
    int width = gridDim.x * blockDim.x;
    int global_id = global_y * width + global_x;

    printf("[2D] 全局ID=%3d, 位置=(%d,%d), 块=(%d,%d), 块内=(%d,%d)\n",
           global_id,
           global_x, global_y,
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y);
}

/*
 * 演示三维网格配置
 */
__global__ void demo_3d_config()
{
    // 计算全局坐标
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_z = blockIdx.z * blockDim.z + threadIdx.z;

    // 计算一维索引
    int width = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;
    int global_id = global_z * width * height + global_y * width + global_x;

    printf("[3D] 全局ID=%4d, 位置=(%d,%d,%d)\n",
           global_id, global_x, global_y, global_z);
}

// ==================== __host__ 和混合限定符示例 ====================

/*
 * __host__ 函数说明：
 * ===================
 *
 * - 在 CPU 上执行
 * - 这是默认的函数类型（如果不指定限定符）
 * - 只能从 CPU 调用
 *
 * 注意：单独使用 __host__ 是可选的，因为这是默认行为
 */
__host__ void print_separator(const char *title)
{
    printf("\n=========================================\n");
    printf("   %s\n", title);
    printf("=========================================\n\n");
}

/*
 * 普通的 CPU 函数（等同于 __host__ void cpu_function()）
 */
void cpu_function()
{
    printf("这是一个 CPU 函数\n");
}

/*
 * __host__ __device__ 混合限定符：
 * =================================
 *
 * 函数可以在 CPU 和 GPU 上都执行
 * 对于一些通用计算很有用
 */
__host__ __device__ int square(int x)
{
    return x * x;
}

/*
 * 在内核中调用混合限定符函数
 */
__global__ void demo_hybrid_function(int *output)
{
    int idx = threadIdx.x;
    output[idx] = square(idx);  // 调用 __host__ __device__ 函数
    printf("线程 %d: square(%d) = %d\n", idx, idx, output[idx]);
}

// ==================== 主函数 ====================

int main()
{
    printf("=========================================================\n");
    printf("   第七章：CUDA 内核深入理解\n");
    printf("=========================================================\n");

    // ==================== 第一部分：函数限定符演示 ====================
    print_separator("第一部分：__device__ 函数演示");

    printf("__device__ 函数特点:\n");
    printf("  - 在 GPU 上执行\n");
    printf("  - 只能被 GPU 函数调用\n");
    printf("  - 用于封装 GPU 端的通用逻辑\n\n");

    // 启动内核演示 __device__ 函数
    printf("启动配置: <<<2, 4>>> (2 个块, 每块 4 个线程)\n\n");
    demo_device_function<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\n");

    // 演示 __device__ 函数链式调用
    printf("演示 __device__ 函数链式调用（阶乘计算）:\n");
    printf("启动配置: <<<1, 6>>> (1 个块, 6 个线程)\n\n");
    demo_device_complex<<<1, 6>>>();
    cudaDeviceSynchronize();

    // ==================== 第二部分：一维配置演示 ====================
    print_separator("第二部分：一维网格配置演示");

    printf("一维网格配置语法: kernel<<<grid, block>>>(args)\n");
    printf("  grid: 块的数量 (int 或 dim3)\n");
    printf("  block: 每块的线程数 (int 或 dim3)\n\n");

    // 示例：不同的启动配置
    int n = 16;
    int *d_output;
    cudaMalloc(&d_output, n * sizeof(int));

    printf("配置 1: <<<1, 16>>> - 1 个块, 每块 16 个线程\n");
    demo_1d_config<<<1, 16>>>(d_output, n);
    cudaDeviceSynchronize();

    printf("\n配置 2: <<<2, 8>>> - 2 个块, 每块 8 个线程\n");
    demo_1d_config<<<2, 8>>>(d_output, n);
    cudaDeviceSynchronize();

    printf("\n配置 3: <<<4, 4>>> - 4 个块, 每块 4 个线程\n");
    demo_1d_config<<<4, 4>>>(d_output, n);
    cudaDeviceSynchronize();

    cudaFree(d_output);

    // ==================== 第三部分：二维配置演示 ====================
    print_separator("第三部分：二维网格配置演示");

    printf("二维网格配置语法:\n");
    printf("  dim3 grid(width, height);    // 网格大小\n");
    printf("  dim3 block(width, height);  // 块大小\n");
    printf("  kernel<<<grid, block>>>(args);\n\n");

    printf("配置: dim3 grid(2, 2), dim3 block(2, 2)\n");
    printf("  网格: 2x2 = 4 个块\n");
    printf("  每块: 2x2 = 4 个线程\n");
    printf("  总线程数: 4 * 4 = 16\n\n");

    dim3 grid_2d(2, 2);
    dim3 block_2d(2, 2);
    demo_2d_config<<<grid_2d, block_2d>>>();
    cudaDeviceSynchronize();

    // ==================== 第四部分：三维配置演示 ====================
    print_separator("第四部分：三维网格配置演示");

    printf("三维网格配置语法:\n");
    printf("  dim3 grid(x, y, z);    // 网格大小\n");
    printf("  dim3 block(x, y, z);  // 块大小\n\n");

    printf("配置: dim3 grid(2, 2, 2), dim3 block(2, 1, 1)\n");
    printf("  网格: 2x2x2 = 8 个块\n");
    printf("  每块: 2x1x1 = 2 个线程\n");
    printf("  总线程数: 8 * 2 = 16\n\n");

    dim3 grid_3d(2, 2, 2);
    dim3 block_3d(2, 1, 1);
    demo_3d_config<<<grid_3d, block_3d>>>();
    cudaDeviceSynchronize();

    // ==================== 第五部分：混合限定符演示 ====================
    print_separator("第五部分：__host__ __device__ 混合限定符演示");

    printf("__host__ __device__ 混合限定符:\n");
    printf("  - 函数可以在 CPU 和 GPU 上都执行\n");
    printf("  - 对于通用计算很有用\n\n");

    // 在 CPU 上调用
    printf("在 CPU 上调用 square() 函数:\n");
    for (int i = 0; i < 5; i++)
    {
        printf("  square(%d) = %d\n", i, square(i));
    }

    // 在 GPU 上调用
    printf("\n在 GPU 上调用 square() 函数:\n");
    printf("启动配置: <<<1, 5>>> (1 个块, 5 个线程)\n\n");

    int *d_squares;
    cudaMalloc(&d_squares, 5 * sizeof(int));
    demo_hybrid_function<<<1, 5>>>(d_squares);
    cudaDeviceSynchronize();
    cudaFree(d_squares);

    // ==================== 总结 ====================
    print_separator("学习要点总结");

    printf("1. 函数限定符:\n");
    printf("   - __global__: 内核函数，GPU 执行，CPU 调用\n");
    printf("   - __device__: 设备函数，GPU 执行，GPU 调用\n");
    printf("   - __host__: 主机函数，CPU 执行，CPU 调用（默认）\n");
    printf("   - __host__ __device__: 可在两边调用\n\n");

    printf("2. 内核启动配置:\n");
    printf("   一维: kernel<<<grid, block>>>(args)\n");
    printf("   二维: kernel<<<dim3(gx,gy), dim3(bx,by)>>>(args)\n");
    printf("   三维: kernel<<<dim3(gx,gy,gz), dim3(bx,by,bz)>>>(args)\n\n");

    printf("3. 索引计算公式:\n");
    printf("   一维: global_id = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("   二维: global_x = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("         global_y = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("   三维: global_x = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("         global_y = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("         global_z = blockIdx.z * blockDim.z + threadIdx.z\n\n");

    printf("4. 硬件限制:\n");
    printf("   - 每块最大线程数: 1024\n");
    printf("   - 块各维度最大值: (1024, 1024, 64)\n");
    printf("   - blockDim.x * blockDim.y * blockDim.z <= 1024\n\n");

    printf("=========================================================\n");
    printf("   恭喜！你已经掌握了 CUDA 内核的核心概念！\n");
    printf("=========================================================\n");

    return 0;
}