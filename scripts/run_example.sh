#!/bin/bash
# ============================================================================
# CUDA 示例运行脚本
# 功能：快速运行对应章节的示例代码
# 用法：./run_example.sh <章节号> [参数...]
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(dirname "$SCRIPT_DIR")/examples/build"

# 示例配置 (章节号 => 目录名:可执行文件名:描述)
declare -A EXAMPLES=(
    ["1"]="01_gpu_basics:cpu_vs_gpu:GPU 并行计算基础 - CPU vs GPU 性能对比"
    ["2"]="02_cuda_intro:cuda_basics:CUDA 基础概念 - 函数类型演示"
    ["3"]="03_gpu_hardware:device_info:GPU 硬件架构 - 设备属性查询"
    ["4"]="04_thread_hierarchy:thread_index:线程层级结构 - 索引计算演示"
    ["5"]="05_hello_cuda:hello_cuda:第一个 CUDA 程序 - Hello GPU"
    ["6"]="06_memory_basics:memory_demo:内存管理基础 - cudaMalloc/cudaMemcpy 演示"
    ["7"]="07_kernel_deep:kernel_demo:核函数深入 - 启动配置演示"
    ["8"]="08_profiling:profiling_demo:性能分析入门 - CUDA 事件计时"
    ["9"]="09_memory_opt:memory_opt:内存访问优化 - 合并访问与向量化"
    ["10"]="10_precision:precision_demo:精度与性能 - FP32 vs FP16 对比"
    ["11"]="11_roofline:roofline_demo:Roofline 模型 - 计算强度分析"
)

# ============================================================================
# 显示帮助
# ============================================================================
show_help() {
    echo -e "${CYAN}CUDA 示例运行脚本${NC}"
    echo ""
    echo "用法: $0 <章节号> [参数...]"
    echo ""
    echo "可用的章节号:"
    echo "-------------------------------------------"

    for chapter in $(echo "${!EXAMPLES[@]}" | tr ' ' '\n' | sort -n); do
        IFS=':' read -r dir exe desc <<< "${EXAMPLES[$chapter]}"
        printf "  ${GREEN}%2s${NC}  %s\n" "$chapter" "$desc"
    done

    echo ""
    echo "选项:"
    echo "  --list, -l     列出所有可执行文件"
    echo "  --all, -a      运行所有示例"
    echo "  --help, -h     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 5            # 运行第 5 章示例"
    echo "  $0 1 1024       # 运行第 1 章示例，传递参数 1024"
    echo "  $0 --all        # 运行所有示例"
}

# ============================================================================
# 列出所有可执行文件
# ============================================================================
list_executables() {
    echo -e "${CYAN}已编译的可执行文件:${NC}"
    echo ""

    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}构建目录不存在，请先运行 ./compile_all.sh${NC}"
        return 1
    fi

    for chapter in $(echo "${!EXAMPLES[@]}" | tr ' ' '\n' | sort -n); do
        IFS=':' read -r dir exe desc <<< "${EXAMPLES[$chapter]}"
        local exe_path="$BUILD_DIR/$dir/$exe"

        if [ -f "$exe_path" ] && [ -x "$exe_path" ]; then
            local size=$(ls -lh "$exe_path" | awk '{print $5}')
            printf "  ${GREEN}%2s${NC}  %-25s %s\n" "$chapter" "$exe" "[$size]"
        else
            printf "  ${RED}%2s${NC}  %-25s (未编译)\n" "$chapter" "$exe"
        fi
    done

    echo ""
}

# ============================================================================
# 运行单个示例
# ============================================================================
run_example() {
    local chapter=$1
    shift

    # 检查章节号是否有效
    if [ -z "${EXAMPLES[$chapter]}" ]; then
        echo -e "${RED}错误: 无效的章节号 '$chapter'${NC}"
        echo ""
        show_help
        exit 1
    fi

    IFS=':' read -r dir exe desc <<< "${EXAMPLES[$chapter]}"
    local exe_path="$BUILD_DIR/$dir/$exe"

    # 检查可执行文件是否存在
    if [ ! -f "$exe_path" ]; then
        echo -e "${RED}错误: 可执行文件不存在${NC}"
        echo "路径: $exe_path"
        echo ""
        echo -e "${YELLOW}请先运行 ./compile_all.sh 编译示例${NC}"
        exit 1
    fi

    # 运行示例
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  第 $chapter 章: $desc${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""
    echo -e "${BLUE}执行: $exe_path $*${NC}"
    echo ""

    "$exe_path" "$@"

    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[✓] 示例运行成功${NC}"
    else
        echo -e "${RED}[✗] 示例运行失败 (退出码: $exit_code)${NC}"
    fi

    return $exit_code
}

# ============================================================================
# 运行所有示例
# ============================================================================
run_all() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  运行所有示例${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""

    local success=0
    local failed=0

    for chapter in $(echo "${!EXAMPLES[@]}" | tr ' ' '\n' | sort -n); do
        if run_example "$chapter"; then
            ((success++))
        else
            ((failed++))
        fi
        echo ""
        echo "-------------------------------------------"
        echo ""
    done

    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  运行结果汇总${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""
    echo -e "成功: ${GREEN}$success${NC}"
    echo -e "失败: ${RED}$failed${NC}"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    # 检查构建目录
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}构建目录不存在${NC}"
        echo -e "${YELLOW}请先运行: ./compile_all.sh${NC}"
        echo ""
        exit 1
    fi

    case "${1:-}" in
        -h|--help)
            show_help
            ;;
        -l|--list)
            list_executables
            ;;
        -a|--all)
            run_all
            ;;
        *)
            if [ -z "${1:-}" ]; then
                show_help
                exit 1
            fi
            run_example "$@"
            ;;
    esac
}

main "$@"