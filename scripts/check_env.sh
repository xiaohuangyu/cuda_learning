#!/bin/bash
# ============================================================================
# CUDA 环境检查脚本
# 功能：检查 CUDA 开发环境是否正确配置
# 用法：./check_env.sh
# ============================================================================

echo "=========================================="
echo "       CUDA 环境检查工具"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# 检查计数
PASS_COUNT=0
FAIL_COUNT=0

# 辅助函数：打印检查结果
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}[✓]${NC} $2"
        ((PASS_COUNT++))
    else
        echo -e "${RED}[✗]${NC} $2"
        ((FAIL_COUNT++))
    fi
}

# 辅助函数：打印警告信息
print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# 辅助函数：打印信息
print_info() {
    echo -e "    $1"
}

echo "-------------------------------------------"
echo "1. 检查 NVIDIA 驱动"
echo "-------------------------------------------"

# 检查 nvidia-smi 是否存在
if command -v nvidia-smi &> /dev/null; then
    print_result 0 "nvidia-smi 命令存在"

    # 获取驱动版本
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$DRIVER_VERSION" ]; then
        print_info "驱动版本: $DRIVER_VERSION"
    fi

    # 获取 GPU 信息
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        print_info "GPU 设备: $GPU_NAME"
    fi

    # 获取显存信息
    MEMORY_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$MEMORY_TOTAL" ]; then
        print_info "显存大小: $MEMORY_TOTAL"
    fi
else
    print_result 1 "nvidia-smi 命令不存在"
    print_warning "请安装 NVIDIA 驱动"
fi

echo ""
echo "-------------------------------------------"
echo "2. 检查 CUDA 编译器 (nvcc)"
echo "-------------------------------------------"

# 检查 nvcc 是否存在
if command -v nvcc &> /dev/null; then
    print_result 0 "nvcc 编译器存在"

    # 获取 CUDA 版本
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    if [ -n "$CUDA_VERSION" ]; then
        print_info "CUDA 版本: $CUDA_VERSION"
    fi

    # 获取 nvcc 完整版本信息
    NVCC_VERSION=$(nvcc --version | grep "nvcc" | head -1)
    if [ -n "$NVCC_VERSION" ]; then
        print_info "编译器版本: $NVCC_VERSION"
    fi
else
    print_result 1 "nvcc 编译器不存在"
    print_warning "请安装 CUDA Toolkit 并添加到 PATH"
fi

echo ""
echo "-------------------------------------------"
echo "3. 检查环境变量"
echo "-------------------------------------------"

# 检查 CUDA_PATH
if [ -n "$CUDA_PATH" ]; then
    print_result 0 "CUDA_PATH 已设置"
    print_info "CUDA_PATH = $CUDA_PATH"
else
    print_result 1 "CUDA_PATH 未设置"
    print_warning "建议设置 CUDA_PATH 环境变量"
fi

# 检查 PATH 中的 CUDA 路径
if [[ "$PATH" == *"cuda"* ]] || [[ "$PATH" == *"CUDA"* ]]; then
    print_result 0 "PATH 中包含 CUDA 路径"
else
    print_warning "PATH 中可能未包含 CUDA bin 目录"
fi

# 检查 LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    if [[ "$LD_LIBRARY_PATH" == *"cuda"* ]] || [[ "$LD_LIBRARY_PATH" == *"CUDA"* ]]; then
        print_result 0 "LD_LIBRARY_PATH 已配置 CUDA 库路径"
    else
        print_warning "LD_LIBRARY_PATH 可能未包含 CUDA 库路径"
    fi
else
    print_warning "LD_LIBRARY_PATH 未设置"
fi

echo ""
echo "-------------------------------------------"
echo "4. 检查 GPU 计算能力"
echo "-------------------------------------------"

# 使用 nvidia-smi 获取计算能力
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$COMPUTE_CAP" ] && [ "$COMPUTE_CAP" != "N/A" ]; then
        print_result 0 "成功获取 GPU 计算能力"
        print_info "计算能力: $COMPUTE_CAP"

        # 根据计算能力给出建议
        MAJOR_CAP=$(echo $COMPUTE_CAP | cut -d'.' -f1)
        if [ "$MAJOR_CAP" -ge 7 ]; then
            print_info "架构建议: sm_$COMPUTE_CAP (支持 Tensor Core)"
        else
            print_info "架构建议: sm_$COMPUTE_CAP"
        fi
    else
        print_warning "无法通过 nvidia-smi 获取计算能力"
        print_info "可以使用 deviceQuery 工具获取详细信息"
    fi
else
    print_result 1 "无法检查 GPU 计算能力"
fi

echo ""
echo "-------------------------------------------"
echo "5. 检查 CUDA 库文件"
echo "-------------------------------------------"

# 常见 CUDA 库检查
CUDA_LIBS=("libcudart.so" "libcurand.so" "libcublas.so" "libcufft.so")
CUDA_LIB_PATHS=("/usr/local/cuda/lib64" "/usr/local/cuda/lib" "/usr/lib/x86_64-linux-gnu")

for lib in "${CUDA_LIBS[@]}"; do
    FOUND=0
    for path in "${CUDA_LIB_PATHS[@]}"; do
        if [ -f "$path/$lib" ]; then
            print_result 0 "找到 $lib 在 $path"
            FOUND=1
            break
        fi
    done
    if [ $FOUND -eq 0 ]; then
        print_warning "未找到 $lib"
    fi
done

echo ""
echo "-------------------------------------------"
echo "6. 测试 CUDA 程序运行"
echo "-------------------------------------------"

# 创建临时测试程序
TEST_DIR=$(mktemp -d)
TEST_FILE="$TEST_DIR/test_cuda.cu"

cat > "$TEST_FILE" << 'EOF'
#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("错误：未检测到支持 CUDA 的 GPU\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU 设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("显存大小: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("SM 数量: %d\n", prop.multiProcessorCount);

    hello_cuda<<<1, 3>>>();
    cudaDeviceSynchronize();

    printf("CUDA 程序运行成功！\n");
    return 0;
}
EOF

# 尝试编译和运行
if command -v nvcc &> /dev/null; then
    print_info "正在编译测试程序..."

    # 获取计算能力用于编译
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$COMPUTE_CAP" ] && [ "$COMPUTE_CAP" != "N/A" ]; then
        ARCH_FLAG="-arch=sm_${COMPUTE_CAP//./}"
    else
        # 默认使用 sm_60 兼容大多数现代 GPU
        ARCH_FLAG="-arch=sm_60"
    fi

    if nvcc $ARCH_FLAG -o "$TEST_DIR/test_cuda" "$TEST_FILE" 2>/dev/null; then
        print_result 0 "测试程序编译成功"

        # 运行测试程序
        print_info "正在运行测试程序..."
        if "$TEST_DIR/test_cuda" &>/dev/null; then
            print_result 0 "CUDA 程序运行成功"
        else
            print_result 1 "CUDA 程序运行失败"
            print_warning "可能需要 root 权限或 GPU 被其他进程占用"
        fi
    else
        print_result 1 "测试程序编译失败"
        print_warning "请检查 CUDA 安装配置"
    fi
else
    print_result 1 "无法测试 CUDA 程序（nvcc 不可用）"
fi

# 清理临时文件
rm -rf "$TEST_DIR"

echo ""
echo "=========================================="
echo "           检查结果汇总"
echo "=========================================="
echo -e "${GREEN}通过: $PASS_COUNT${NC}"
echo -e "${RED}失败: $FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}恭喜！您的 CUDA 环境配置完整。${NC}"
else
    echo -e "${YELLOW}存在 $FAIL_COUNT 个问题需要解决。${NC}"
    echo ""
    echo "常见解决方案："
    echo "1. 安装 NVIDIA 驱动：访问 https://www.nvidia.com/Download/index.aspx"
    echo "2. 安装 CUDA Toolkit：访问 https://developer.nvidia.com/cuda-downloads"
    echo "3. 设置环境变量："
    echo "   export CUDA_PATH=/usr/local/cuda"
    echo "   export PATH=\$CUDA_PATH/bin:\$PATH"
    echo "   export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"
fi

echo ""
echo "检查完成！"