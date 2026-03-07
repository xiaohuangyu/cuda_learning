/*
 * GPU硬件信息查询示例
 * ===================
 * 本程序演示如何使用 cudaDeviceProp 结构体
 * 查询和显示GPU设备的各种硬件属性
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 辅助函数：格式化输出内存大小
// ============================================================================
void print_memory_size(const char *name, size_t bytes) {
    // 自动选择合适的单位显示
    const char *unit;
    double size;

    if (bytes >= 1024ULL * 1024 * 1024) {
        size = (double)bytes / (1024 * 1024 * 1024);
        unit = "GB";
    } else if (bytes >= 1024 * 1024) {
        size = (double)bytes / (1024 * 1024);
        unit = "MB";
    } else if (bytes >= 1024) {
        size = (double)bytes / 1024;
        unit = "KB";
    } else {
        size = (double)bytes;
        unit = "B";
    }

    printf("  %-30s : %.2f %s (%zu 字节)\n", name, size, unit, bytes);
}

// ============================================================================
// 打印设备属性的完整函数
// ============================================================================
void print_device_properties(cudaDeviceProp *prop, int device_id) {
    printf("\n");
    printf("============================================================\n");
    printf("              GPU 设备 %d 详细信息\n", device_id);
    printf("============================================================\n");

    // ------------------------------------------------------------------------
    // 基本信息
    // ------------------------------------------------------------------------
    printf("\n【基本信息】\n");
    printf("  %-30s : %s\n", "设备名称", prop->name);
    printf("  %-30s : %d\n", "计算能力 (Compute Capability)", prop->major * 10 + prop->minor);
    printf("  %-30s : %d.%d\n", "计算能力版本", prop->major, prop->minor);
    printf("  %-30s : %d\n", "设备ID", device_id);

    // ------------------------------------------------------------------------
    // 多处理器信息（SM）
    // ------------------------------------------------------------------------
    printf("\n【多处理器 (SM) 信息】\n");
    printf("  %-30s : %d\n", "SM (Streaming MultiProcessor) 数量", prop->multiProcessorCount);
    printf("  %-30s : %d\n", "每个SM最大驻留线程数", prop->maxThreadsPerMultiProcessor);
    printf("  %-30s : %d\n", "每个SM最大线程块数", prop->maxBlocksPerMultiProcessor);
    printf("  %-30s : %d\n", "每个SM最大warp数", prop->maxThreadsPerMultiProcessor / 32);

    // 计算理论最大线程数
    int max_total_threads = prop->multiProcessorCount * prop->maxThreadsPerMultiProcessor;
    printf("  %-30s : %d\n", "理论最大并发线程数", max_total_threads);

    // ------------------------------------------------------------------------
    // 线程块信息
    // ------------------------------------------------------------------------
    printf("\n【线程块 (Block) 信息】\n");
    printf("  %-30s : %d\n", "每个块最大线程数", prop->maxThreadsPerBlock);
    printf("  %-30s : (%d, %d, %d)\n", "块最大维度", prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);
    printf("  %-30s : (%d, %d, %d)\n", "网格最大维度", prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);

    // ------------------------------------------------------------------------
    // 内存信息
    // ------------------------------------------------------------------------
    printf("\n【内存信息】\n");
    print_memory_size("全局内存总量", prop->totalGlobalMem);
    print_memory_size("共享内存每块", prop->sharedMemPerBlock);
    print_memory_size("共享内存每SM", prop->sharedMemPerMultiprocessor);
    print_memory_size("常量内存总量", prop->totalConstMem);
    printf("  %-30s : %d\n", "每块32位寄存器数量", prop->regsPerBlock);
    printf("  %-30s : %d\n", "每SM 32位寄存器数量", prop->regsPerMultiprocessor);
    printf("  %-30s : %d\n", "内存对齐要求", prop->memPitch);
    printf("  %-30s : %d\n", "纹理对齐要求", prop->textureAlignment);

    // ------------------------------------------------------------------------
    // 时钟和性能信息
    // ------------------------------------------------------------------------
    printf("\n【时钟和性能信息】\n");
    // 注意: clockRate 和 memoryClockRate 在较新 CUDA 版本中已移除
    // 使用 cudaDeviceGetAttribute 获取时钟信息
    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    printf("  %-30s : %d MHz\n", "GPU时钟频率", clock_rate_khz / 1000);

    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    printf("  %-30s : %d MHz\n", "内存时钟频率", mem_clock_khz / 1000);

    printf("  %-30s : %d bits\n", "内存总线宽度", prop->memoryBusWidth);
    printf("  %-30s : %d KB\n", "L2缓存大小", prop->l2CacheSize / 1024);
    printf("  %-30s : %.1f GB/s\n", "理论内存带宽",
           2.0 * mem_clock_khz * (prop->memoryBusWidth / 8) / 1000000.0);

    // ------------------------------------------------------------------------
    // Warp信息
    // ------------------------------------------------------------------------
    printf("\n【Warp 信息】\n");
    printf("  %-30s : %d\n", "Warp大小 (每warp线程数)", prop->warpSize);

    // ------------------------------------------------------------------------
    // 功能支持信息
    // ------------------------------------------------------------------------
    printf("\n【功能支持】\n");
    printf("  %-30s : %s\n", "支持统一内存", prop->unifiedAddressing ? "是" : "否");
    printf("  %-30s : %s\n", "支持并发核函数", prop->concurrentKernels ? "是" : "否");
    printf("  %-30s : %s\n", "支持 ECC 内存", prop->ECCEnabled ? "是" : "否");
    printf("  %-30s : %s\n", "支持协作组", prop->cooperativeLaunch ? "是" : "否");
    printf("  %-30s : %s\n", "支持流优先级", prop->streamPrioritiesSupported ? "是" : "否");
    printf("  %-30s : %d\n", "最大1D纹理大小", prop->maxTexture1D);
    printf("  %-30s : %d\n", "最大表面内存", prop->surfaceAlignment);

    printf("\n============================================================\n");
}

// ============================================================================
// 打印硬件限制总结
// ============================================================================
void print_hardware_limits(cudaDeviceProp *prop) {
    printf("\n");
    printf("============================================================\n");
    printf("              硬件限制速查表\n");
    printf("============================================================\n\n");

    printf("【线程组织限制】\n");
    printf("  最大 Grid 大小 (x方向): %d\n", prop->maxGridSize[0]);
    printf("  最大 Block 大小: %d 线程\n", prop->maxThreadsPerBlock);
    printf("  Warp 大小: %d 线程\n", prop->warpSize);
    printf("  最大 Block 维度: (%d, %d, %d)\n",
           prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);

    printf("\n【内存限制】\n");
    printf("  最大共享内存/Block: %.0f KB\n", prop->sharedMemPerBlock / 1024.0);
    printf("  最大寄存器/Block: %d\n", prop->regsPerBlock);

    printf("\n【计算能力说明】\n");
    printf("  当前设备计算能力: %d.%d\n", prop->major, prop->minor);

    // 根据计算能力给出特性说明
    if (prop->major >= 8) {
        printf("  特性: 支持 Ampere 架构新特性\n");
        printf("        - Tensor Core 第三代\n");
        printf("        - 异步复制\n");
        printf("        - BF16 支持\n");
    } else if (prop->major >= 7) {
        printf("  特性: 支持 Volta/Turing 架构特性\n");
        printf("        - Tensor Core\n");
        printf("        - 独立线程调度\n");
    }

    printf("============================================================\n");
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("========================================\n");
    printf("    GPU 硬件信息查询程序\n");
    printf("========================================\n\n");

    // ------------------------------------------------------------------------
    // 获取设备数量
    // ------------------------------------------------------------------------
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        printf("错误：无法获取CUDA设备数量！\n");
        printf("错误信息：%s\n", cudaGetErrorString(err));
        printf("\n可能的原因：\n");
        printf("  1. 没有安装CUDA驱动\n");
        printf("  2. 没有NVIDIA GPU\n");
        printf("  3. GPU驱动未正确安装\n");
        return -1;
    }

    if (device_count == 0) {
        printf("未检测到支持CUDA的GPU设备。\n");
        return -1;
    }

    printf("检测到 %d 个支持CUDA的GPU设备\n", device_count);
    printf("----------------------------------------\n");

    // ------------------------------------------------------------------------
    // 遍历并打印所有设备信息
    // ------------------------------------------------------------------------
    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);

        if (err != cudaSuccess) {
            printf("错误：无法获取设备 %d 的属性！\n", dev);
            continue;
        }

        // 打印完整的设备属性
        print_device_properties(&prop, dev);
    }

    // ------------------------------------------------------------------------
    // 打印当前设备
    // ------------------------------------------------------------------------
    int current_device;
    cudaGetDevice(&current_device);
    printf("\n当前使用的设备: %d\n", current_device);

    // ------------------------------------------------------------------------
    // 打印硬件限制总结
    // ------------------------------------------------------------------------
    cudaDeviceProp current_prop;
    cudaGetDeviceProperties(&current_prop, current_device);
    print_hardware_limits(&current_prop);

    // ------------------------------------------------------------------------
    // 重要的硬件参数说明
    // ------------------------------------------------------------------------
    printf("\n========================================\n");
    printf("        关键硬件参数说明\n");
    printf("========================================\n\n");

    printf("1. SM (Streaming Multiprocessor) 多处理器:\n");
    printf("   - GPU的核心计算单元\n");
    printf("   - 每个SM可同时执行多个线程块\n");
    printf("   - 更多SM = 更高并行度\n\n");

    printf("2. Warp (线程束):\n");
    printf("   - 32个线程为一组执行\n");
    printf("   - 同一Warp中线程执行相同指令\n");
    printf("   - Warp是GPU调度的基本单位\n\n");

    printf("3. 计算能力 (Compute Capability):\n");
    printf("   - 表示GPU架构版本\n");
    printf("   - 格式：主版本.次版本 (如 8.0)\n");
    printf("   - 决定支持的特性和硬件限制\n\n");

    printf("4. 全局内存 (Global Memory):\n");
    printf("   - GPU的显存\n");
    printf("   - 所有线程都可访问\n");
    printf("   - 带宽是性能关键因素\n\n");

    printf("5. 共享内存 (Shared Memory):\n");
    printf("   - 每个线程块独享的高速缓存\n");
    printf("   - 比全局内存快得多\n");
    printf("   - 用于线程间数据共享\n");

    return 0;
}