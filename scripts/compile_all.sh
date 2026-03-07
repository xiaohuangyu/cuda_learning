#!/bin/bash
# ============================================================================
# CUDA 示例编译脚本 (CMake 版本)
# 功能：使用 CMake 编译所有示例代码
# 用法：./compile_all.sh [选项]
#   选项:
#     --clean       清理构建目录后重新编译
#     --arch=XX     指定 GPU 架构 (如 80, 86, 89, 90)
#     --release     使用 Release 模式编译
#     --help        显示帮助信息
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # 无颜色

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(dirname "$SCRIPT_DIR")/examples"
BUILD_DIR="$EXAMPLES_DIR/build"

# 默认参数
CLEAN_BUILD=false
CUDA_ARCH=""
CUDA_PATH_AUTO=""  # 自动检测的 CUDA 路径
BUILD_TYPE="Debug"

# ============================================================================
# 自动检测 CUDA 安装路径
# ============================================================================
detect_cuda_path() {
    # 优先级: 环境变量 > /usr/local/cuda > 查找最新版本

    # 1. 检查环境变量
    if [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH" ]; then
        CUDA_PATH_AUTO="$CUDA_PATH"
        return 0
    fi

    # 2. 检查默认路径
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH_AUTO="/usr/local/cuda"
        return 0
    fi

    # 3. 查找 cuda-X.Y 格式的目录
    local cuda_dirs=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$cuda_dirs" ]; then
        CUDA_PATH_AUTO="$cuda_dirs"
        return 0
    fi

    return 1
}

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "CUDA 示例编译脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --clean         清理构建目录后重新编译"
    echo "  --arch=XX       指定 GPU 架构 (如 80, 86, 89, 90)"
    echo "  --cuda=PATH     指定 CUDA 安装路径 (默认自动检测)"
    echo "  --release       使用 Release 模式编译 (默认 Debug)"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "GPU 架构参考:"
    echo "  70  - V100 (Volta)"
    echo "  75  - RTX 2080 (Turing)"
    echo "  80  - A100 (Ampere)"
    echo "  86  - RTX 3090 (Ampere)"
    echo "  89  - RTX 4090 (Ada Lovelace)"
    echo "  90  - H100 (Hopper)"
    echo ""
    echo "示例:"
    echo "  $0                            # 使用默认配置编译"
    echo "  $0 --clean                    # 清理后重新编译"
    echo "  $0 --arch=80                  # 指定 GPU 架构"
    echo "  $0 --cuda=/usr/local/cuda-12.0  # 指定 CUDA 版本"
    echo "  $0 --clean --arch=80 --release  # 组合使用"
}

# ============================================================================
# 解析命令行参数
# ============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --arch=*)
                CUDA_ARCH="${1#*=}"
                shift
                ;;
            --cuda=*)
                CUDA_PATH_AUTO="${1#*=}"
                shift
                ;;
            --release)
                BUILD_TYPE="Release"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}错误: 未知选项 '$1'${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# 检查依赖
# ============================================================================
check_dependencies() {
    echo -e "${CYAN}========================================"
    echo "       CUDA 编译环境检查"
    echo -e "========================================${NC}"
    echo ""

    # 自动检测 CUDA 路径（如果用户没有指定）
    if [ -z "$CUDA_PATH_AUTO" ]; then
        detect_cuda_path
    fi

    # 显示使用的 CUDA 路径
    if [ -n "$CUDA_PATH_AUTO" ] && [ -d "$CUDA_PATH_AUTO" ]; then
        echo -e "${GREEN}[✓]${NC} CUDA 路径: $CUDA_PATH_AUTO"
        local nvcc_path="$CUDA_PATH_AUTO/bin/nvcc"
        if [ -f "$nvcc_path" ]; then
            local cuda_ver=$($nvcc_path --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//')
            echo -e "${GREEN}[✓]${NC} CUDA 版本: $cuda_ver"
        fi
    else
        # 使用系统 PATH 中的 nvcc
        if command -v nvcc &> /dev/null; then
            local cuda_ver=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
            echo -e "${GREEN}[✓]${NC} CUDA 版本: $cuda_ver (系统 PATH)"
        else
            echo -e "${RED}[✗] 未找到 nvcc 编译器${NC}"
            echo "    请安装 CUDA Toolkit 或使用 --cuda=PATH 指定路径"
            exit 1
        fi
    fi

    # 检查 cmake
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}[✗] 未找到 cmake${NC}"
        echo "    请安装 cmake: sudo apt install cmake"
        exit 1
    fi

    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    echo -e "${GREEN}[✓]${NC} CMake 版本: $CMAKE_VERSION"

    # 检查 GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        echo -e "${GREEN}[✓]${NC} GPU: $GPU_NAME"
        echo -e "${GREEN}[✓]${NC} 计算能力: $COMPUTE_CAP"
    fi

    echo ""
}

# ============================================================================
# 检测 GPU 架构
# ============================================================================
detect_arch() {
    if [ -n "$CUDA_ARCH" ]; then
        echo -e "${BLUE}使用指定的 GPU 架构: sm_$CUDA_ARCH${NC}"
        return
    fi

    if command -v nvidia-smi &> /dev/null; then
        local cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$cap" ] && [ "$cap" != "N/A" ]; then
            CUDA_ARCH="${cap//./}"
            echo -e "${GREEN}自动检测到 GPU 架构: sm_$CUDA_ARCH${NC}"
            return
        fi
    fi

    # 默认架构
    CUDA_ARCH="80"
    echo -e "${YELLOW}未检测到 GPU，使用默认架构: sm_$CUDA_ARCH${NC}"
}

# ============================================================================
# 清理构建目录
# ============================================================================
clean_build() {
    if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}清理构建目录...${NC}"
        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}[✓] 构建目录已清理${NC}"
        echo ""
    fi
}

# ============================================================================
# 运行 CMake
# ============================================================================
run_cmake() {
    echo -e "${CYAN}========================================"
    echo "       配置 CMake"
    echo -e "========================================${NC}"
    echo ""

    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # CMake 参数
    local cmake_args=(
        "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    )

    # 如果检测到 CUDA 路径，添加到 CMake 参数
    if [ -n "$CUDA_PATH_AUTO" ] && [ -d "$CUDA_PATH_AUTO" ]; then
        cmake_args+=("-DCUDAToolkit_ROOT=$CUDA_PATH_AUTO")
        cmake_args+=("-DCMAKE_CUDA_COMPILER=$CUDA_PATH_AUTO/bin/nvcc")
    fi

    echo "配置命令: cmake .. ${cmake_args[*]}"
    echo ""

    if cmake .. "${cmake_args[@]}"; then
        echo -e "${GREEN}[✓] CMake 配置成功${NC}"
    else
        echo -e "${RED}[✗] CMake 配置失败${NC}"
        exit 1
    fi

    echo ""
}

# ============================================================================
# 编译
# ============================================================================
build() {
    echo -e "${CYAN}========================================"
    echo "       编译示例代码"
    echo -e "========================================${NC}"
    echo ""

    local start_time=$(date +%s)

    # 获取 CPU 核心数
    local num_jobs=$(nproc)
    echo "使用 $num_jobs 个并行任务编译..."
    echo ""

    if make -j$num_jobs; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        echo ""
        echo -e "${GREEN}[✓] 编译成功！耗时: ${duration}s${NC}"
    else
        echo ""
        echo -e "${RED}[✗] 编译失败${NC}"
        exit 1
    fi
}

# ============================================================================
# 显示编译结果
# ============================================================================
show_results() {
    echo ""
    echo -e "${CYAN}========================================"
    echo "       编译结果"
    echo -e "========================================${NC}"
    echo ""

    # 统计可执行文件
    local count=0
    local executables=()

    for dir in "$BUILD_DIR"/*_*/; do
        if [ -d "$dir" ]; then
            for exe in "$dir"*; do
                if [ -f "$exe" ] && [ -x "$exe" ]; then
                    executables+=("$exe")
                    ((count++))
                fi
            done
        fi
    done

    echo -e "编译成功: ${GREEN}$count${NC} 个可执行文件"
    echo ""

    if [ ${#executables[@]} -gt 0 ]; then
        echo "可执行文件列表:"
        echo "-------------------------------------------"

        local i=1
        for exe in "${executables[@]}"; do
            local name=$(basename "$exe")
            local size=$(ls -lh "$exe" | awk '{print $5}')
            printf "  %2d. %-20s %s\n" "$i" "$name" "$size"
            ((i++))
        done

        echo ""
    fi

    # 显示运行提示
    echo -e "${CYAN}========================================"
    echo "       运行示例"
    echo -e "========================================${NC}"
    echo ""
    echo "运行示例命令:"
    echo ""
    echo "  cd $BUILD_DIR"
    echo ""
    echo "  # 第一章: GPU 基础"
    echo "  ./01_gpu_basics/cpu_vs_gpu"
    echo ""
    echo "  # 第五章: 第一个程序"
    echo "  ./05_hello_cuda/hello_cuda"
    echo ""
    echo "  # 第三章: GPU 硬件信息"
    echo "  ./03_gpu_hardware/device_info"
    echo ""
    echo "  # 第十一章: Roofline 模型"
    echo "  ./11_roofline/roofline_demo"
    echo ""
    echo "-------------------------------------------"
    echo "提示: 运行 ./run_example.sh <章节号> 可快速运行对应示例"
    echo ""
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    parse_args "$@"

    echo -e "${CYAN}"
    echo "=========================================="
    echo "     CUDA 学习示例 - 编译脚本"
    echo "=========================================="
    echo -e "${NC}"
    echo ""

    check_dependencies
    detect_arch
    clean_build
    run_cmake
    build
    show_results
}

# 运行主函数
main "$@"