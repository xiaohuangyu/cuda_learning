/*
 * CUDA线程层次结构示例
 * =======================
 * 本程序演示CUDA的线程层次结构：Grid（网格）→ Block（线程块）→ Thread（线程）
 * 展示如何使用blockIdx、threadIdx计算全局索引，以及1D和2D索引方式
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 示例1：一维线程索引
// ============================================================================
/**
 * 一维线程索引的核函数
 * 展示最简单的线程索引计算方式
 *
 * 一维情况下：
 * - Grid是一维的线程块数组
 * - Block是一维的线程数组
 * - 全局索引 = blockIdx.x * blockDim.x + threadIdx.x
 */
__global__ void kernel_1d_indexing(int *output, int n) {
    // 计算全局索引
    // blockIdx.x: 当前线程块在网格中的索引（0开始）
    // blockDim.x: 每个线程块中的线程数量
    // threadIdx.x: 当前线程在线程块中的索引（0开始）
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：避免越界访问
    if (global_idx < n) {
        output[global_idx] = global_idx;
    }
}

// ============================================================================
// 示例2：打印线程信息（一维）
// ============================================================================
/**
 * 打印一维线程信息的核函数
 * 用于理解线程层次结构
 */
__global__ void kernel_print_1d_info() {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 注意：printf在核函数中输出顺序是未定义的
    // 因为多个线程并行执行
    printf("线程 %d: blockIdx.x=%d, threadIdx.x=%d, blockDim.x=%d\n",
           global_idx, blockIdx.x, threadIdx.x, blockDim.x);
}

// ============================================================================
// 示例3：二维线程索引
// ============================================================================
/**
 * 二维线程索引的核函数
 * 展示如何处理二维数据（如矩阵）
 *
 * 二维情况下：
 * - Grid是二维的线程块矩阵
 * - Block是二维的线程矩阵
 * - 需要分别计算行和列索引
 */
__global__ void kernel_2d_indexing(int *matrix, int width, int height) {
    // 计算列索引（x方向）
    // blockIdx.x: 当前块在网格中的列索引
    // blockDim.x: 每块在x方向的线程数
    // threadIdx.x: 当前线程在块中的列索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算行索引（y方向）
    // blockIdx.y: 当前块在网格中的行索引
    // blockDim.y: 每块在y方向的线程数
    // threadIdx.y: 当前线程在块中的行索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (row < height && col < width) {
        // 将二维索引转换为一维索引
        int global_idx = row * width + col;
        matrix[global_idx] = global_idx;
    }
}

// ============================================================================
// 示例4：打印二维线程信息
// ============================================================================
/**
 * 打印二维线程信息的核函数
 */
__global__ void kernel_print_2d_info() {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    printf("线程 (%d,%d): blockIdx=(%d,%d), threadIdx=(%d,%d), blockDim=(%d,%d)\n",
           row, col,
           blockIdx.y, blockIdx.x,  // 注意：y是行，x是列
           threadIdx.y, threadIdx.x,
           blockDim.y, blockDim.x);
}

// ============================================================================
// 示例5：计算全局线程数量
// ============================================================================
/**
 * 展示如何获取当前Grid中的总线程数
 */
__global__ void kernel_grid_info() {
    // gridDim: 网格维度（包含的块数量）
    // blockDim: 块维度（每块包含的线程数量）

    // 计算当前Grid的总线程数
    int total_threads_x = gridDim.x * blockDim.x;
    int total_threads_y = gridDim.y * blockDim.y;
    int total_threads = total_threads_x * total_threads_y;

    // 只让第一个线程打印（避免输出混乱）
    if (blockIdx.x == 0 && blockIdx.y == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Grid信息: gridDim=(%d,%d,%d)\n",
               gridDim.x, gridDim.y, gridDim.z);
        printf("Block信息: blockDim=(%d,%d,%d)\n",
               blockDim.x, blockDim.y, blockDim.z);
        printf("总线程数: %d\n", total_threads);
    }
}

// ============================================================================
// 示例6：三维线程索引（扩展）
// ============================================================================
/**
 * 三维线程索引的核函数示例
 * 用于处理三维数据（如体积数据）
 */
__global__ void kernel_3d_indexing(int *volume, int dim_x, int dim_y, int dim_z) {
    // 三维索引计算
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // 边界检查
    if (x < dim_x && y < dim_y && z < dim_z) {
        // 三维转一维索引
        int global_idx = z * dim_y * dim_x + y * dim_x + x;
        volume[global_idx] = global_idx;
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("========================================\n");
    printf("    CUDA 线程层次结构演示程序\n");
    printf("========================================\n\n");

    // ========================================================================
    // 线程层次结构说明
    // ========================================================================
    printf("【CUDA 线程层次结构】\n\n");
    printf("Grid (网格)\n");
    printf("  │\n");
    printf("  ├── Block(0,0)     Block(1,0)     Block(2,0)  ...\n");
    printf("  │     │               │               │\n");
    printf("  │   Thread         Thread          Thread\n");
    printf("  │     │               │               │\n");
    printf("  │   (0,0)..(n,m)   (0,0)..(n,m)    (0,0)..(n,m)\n");
    printf("  │\n");
    printf("  ├── Block(0,1)     Block(1,1)     ...\n");
    printf("  │\n");
    printf("  └── ...\n\n");

    // ========================================================================
    // 第一部分：一维索引演示
    // ========================================================================
    printf("========================================\n");
    printf("        第一部分：一维索引\n");
    printf("========================================\n\n");

    // 配置参数
    const int N_1D = 16;          // 总数据量
    int block_size_1d = 4;        // 每块线程数
    int grid_size_1d = (N_1D + block_size_1d - 1) / block_size_1d;  // 向上取整

    printf("【配置】\n");
    printf("  数据总量: %d\n", N_1D);
    printf("  线程块数量 (gridDim.x): %d\n", grid_size_1d);
    printf("  每块线程数 (blockDim.x): %d\n", block_size_1d);
    printf("  总线程数: %d\n\n", grid_size_1d * block_size_1d);

    printf("【线程信息】\n");
    kernel_print_1d_info<<<grid_size_1d, block_size_1d>>>();
    cudaDeviceSynchronize();

    printf("\n【一维索引公式】\n");
    printf("  global_idx = blockIdx.x * blockDim.x + threadIdx.x\n\n");

    // 验证结果
    int *h_output_1d = new int[N_1D];
    int *d_output_1d;
    cudaMalloc(&d_output_1d, N_1D * sizeof(int));

    kernel_1d_indexing<<<grid_size_1d, block_size_1d>>>(d_output_1d, N_1D);
    cudaMemcpy(h_output_1d, d_output_1d, N_1D * sizeof(int), cudaMemcpyDeviceToHost);

    printf("【计算结果】\n  ");
    for (int i = 0; i < N_1D; i++) {
        printf("%d ", h_output_1d[i]);
    }
    printf("\n");

    cudaFree(d_output_1d);
    delete[] h_output_1d;

    // ========================================================================
    // 第二部分：二维索引演示
    // ========================================================================
    printf("\n========================================\n");
    printf("        第二部分：二维索引\n");
    printf("========================================\n\n");

    const int WIDTH = 8;
    const int HEIGHT = 6;

    // 二维配置：块大小为4x2
    dim3 block_2d(4, 2);  // 每块4列2行
    dim3 grid_2d((WIDTH + block_2d.x - 1) / block_2d.x,
                 (HEIGHT + block_2d.y - 1) / block_2d.y);

    printf("【配置】\n");
    printf("  矩阵大小: %d x %d = %d 元素\n", WIDTH, HEIGHT, WIDTH * HEIGHT);
    printf("  线程块维度 (blockDim): (%d, %d)\n", block_2d.x, block_2d.y);
    printf("  网格维度 (gridDim): (%d, %d)\n", grid_2d.x, grid_2d.y);
    printf("  每块线程数: %d\n", block_2d.x * block_2d.y);
    printf("  总线程数: %d\n\n", grid_2d.x * grid_2d.y * block_2d.x * block_2d.y);

    printf("【二维索引公式】\n");
    printf("  col = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("  row = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("  global_idx = row * width + col\n\n");

    // 验证二维索引
    int *h_matrix = new int[WIDTH * HEIGHT];
    int *d_matrix;
    cudaMalloc(&d_matrix, WIDTH * HEIGHT * sizeof(int));

    kernel_2d_indexing<<<grid_2d, block_2d>>>(d_matrix, WIDTH, HEIGHT);
    cudaMemcpy(h_matrix, d_matrix, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    printf("【矩阵结果】\n");
    printf("     ");
    for (int c = 0; c < WIDTH; c++) printf("%6d", c);
    printf("\n");
    for (int r = 0; r < HEIGHT; r++) {
        printf("  %d: ", r);
        for (int c = 0; c < WIDTH; c++) {
            printf("%6d", h_matrix[r * WIDTH + c]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    delete[] h_matrix;

    // ========================================================================
    // 第三部分：线程块映射可视化
    // ========================================================================
    printf("\n========================================\n");
    printf("      第三部分：线程块映射可视化\n");
    printf("========================================\n\n");

    printf("【一维示例：4块，每块4线程】\n\n");
    printf("  Grid: [Block0] [Block1] [Block2] [Block3]\n");
    printf("          ││││    ││││    ││││    ││││\n");
    printf("          TTTT    TTTT    TTTT    TTTT\n");
    printf("          0123    4567    89..    ..15\n");
    printf("          \n");
    printf("  blockIdx.x:  0       1       2       3\n");
    printf("  threadIdx.x: 0-3     0-3     0-3     0-3\n");
    printf("  globalIdx:   0-3     4-7     8-11    12-15\n\n");

    printf("【二维示例：2x2 Grid, 2x2 Block】\n\n");
    printf("         Grid\n");
    printf("         ┌────────────┬────────────┐\n");
    printf("         │ Block(0,0) │ Block(1,0) │\n");
    printf("         │ T00 T10    │ T00 T10    │\n");
    printf("         │ T01 T11    │ T01 T11    │\n");
    printf("         ├────────────┼────────────┤\n");
    printf("         │ Block(0,1) │ Block(1,1) │\n");
    printf("         │ T00 T10    │ T00 T10    │\n");
    printf("         │ T01 T11    │ T01 T11    │\n");
    printf("         └────────────┴────────────┘\n\n");

    // ========================================================================
    // 第四部分：内置变量说明
    // ========================================================================
    printf("========================================\n");
    printf("        第四部分：内置变量说明\n");
    printf("========================================\n\n");

    printf("【在核函数中可用的内置变量】\n\n");

    printf("1. blockIdx (uint3类型)\n");
    printf("   - 当前线程块在网格中的索引\n");
    printf("   - blockIdx.x, blockIdx.y, blockIdx.z\n\n");

    printf("2. threadIdx (uint3类型)\n");
    printf("   - 当前线程在线程块中的索引\n");
    printf("   - threadIdx.x, threadIdx.y, threadIdx.z\n\n");

    printf("3. blockDim (dim3类型)\n");
    printf("   - 线程块的维度（每块线程数）\n");
    printf("   - blockDim.x, blockDim.y, blockDim.z\n\n");

    printf("4. gridDim (dim3类型)\n");
    printf("   - 网格的维度（块数量）\n");
    printf("   - gridDim.x, gridDim.y, gridDim.z\n\n");

    printf("【索引计算总结】\n");
    printf("  一维: idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("  二维: row = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("        col = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("        idx = row * width + col\n");
    printf("  三维: x = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("        y = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("        z = blockIdx.z * blockDim.z + threadIdx.z\n");
    printf("        idx = z * height * width + y * width + x\n");

    // ========================================================================
    // 第五部分：dim3类型说明
    // ========================================================================
    printf("\n========================================\n");
    printf("        第五部分：dim3 类型\n");
    printf("========================================\n\n");

    printf("【dim3 类型说明】\n");
    printf("  CUDA用于定义维度的内置类型\n\n");

    printf("  用法示例：\n");
    printf("    dim3 block(16, 16);     // 2D: 16x16 线程\n");
    printf("    dim3 block(16, 16, 1);  // 同上，z=1\n");
    printf("    dim3 block(256);        // 1D: 256 线程 (等同于 dim3(256,1,1))\n\n");

    printf("  网格配置：\n");
    printf("    dim3 grid((width + block.x - 1) / block.x,\n");
    printf("              (height + block.y - 1) / block.y);\n\n");

    printf("【常用配置示例】\n");
    printf("  1D: kernel<<<numBlocks, threadsPerBlock>>>(...)\n");
    printf("  2D: kernel<<<grid, block>>>(...)\n");
    printf("  3D: kernel<<<dim3(gx,gy,gz), dim3(bx,by,bz)>>>(...)\n");

    return 0;
}