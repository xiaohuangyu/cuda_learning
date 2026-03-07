# 第三章：GPU 硬件架构入门

> 学习目标：了解 GPU 内部结构，理解 SM、CUDA Core、Warp 等核心概念
>
> 预计阅读时间：25 分钟
>
> 前置知识：[第二章：什么是 CUDA？](./02_CUDA是什么.md)

---

## 1. 从宏观到微观：GPU 架构总览

### 1.1 GPU 内部结构图

```mermaid
graph TB
    subgraph GPU["NVIDIA GPU 内部结构"]
        subgraph SMs["流多处理器组 (SMs)"]
            SM0["SM 0"]
            SM1["SM 1"]
            SM2["SM 2"]
            SMn["..."]
            SM108["SM 107"]
        end

        subgraph Memory["内存子系统"]
            L2["L2 Cache<br/>(共享)"]
            HBM["HBM 显存<br/>(Global Memory)"]
        end

        SM0 --> L2
        SM1 --> L2
        SM2 --> L2
        SMn --> L2
        SM108 --> L2
        L2 --> HBM
    end

    style GPU fill:#F5F5F5
    style SMs fill:#E3F2FD
    style Memory fill:#FFF3E0
```

### 1.2 NVIDIA A100 参数示例

| 组件 | 数量 | 说明 |
|------|------|------|
| **SM（流多处理器）** | 108 个 | GPU 的基本计算单元 |
| **CUDA Core** | 6912 个 | 每个 SM 有 64 个 |
| **Tensor Core** | 432 个 | 矩阵运算加速单元 |
| **显存（HBM2e）** | 80 GB | 全局内存 |
| **带宽** | 2039 GB/s | 内存带宽 |

---

## 2. SM（流多处理器）详解

### 2.1 什么是 SM？

**SM**（Streaming Multiprocessor，流多处理器）是 GPU 的基本计算单元。

### 2.2 一个类比

把 GPU 想象成一个大型工厂：

```mermaid
graph LR
    subgraph 工厂类比["工厂类比"]
        A["GPU = 整个工厂"]
        B["SM = 一个车间"]
        C["CUDA Core = 一个工人"]
        D["Warp = 一个工作组（32人）"]
    end

    A --> B --> C
    B --> D
```

| 概念 | 类比 | 说明 |
|------|------|------|
| **GPU** | 整个工厂 | 所有计算资源的集合 |
| **SM** | 一个车间 | 独立的计算单元 |
| **CUDA Core** | 一个工人 | 执行基本运算 |
| **Warp** | 工作组 | 协同工作的 32 个线程 |

### 2.3 SM 内部结构

```mermaid
graph TB
    subgraph SM["SM（流多处理器）内部结构"]
        subgraph Schedulers["调度器"]
            WS["Warp 调度器"]
            DS["分发单元"]
        end

        subgraph Cores["计算单元"]
            direction LR
            CC0["CUDA Core<br/>FP32"]
            CC1["CUDA Core<br/>INT32"]
            TC["Tensor Core<br/>矩阵运算"]
            SFU["特殊功能单元<br/>(SFU)"]
        end

        subgraph Memory["内存资源"]
            Reg["寄存器堆<br/>(Register File)<br/>最快"]
            Shared["共享内存<br/>(Shared Memory)"]
            L1["L1 Cache"]
        end
    end

    WS --> DS --> Cores
    Cores --> Memory
```

### 2.4 SM 的工作原理

```mermaid
sequenceDiagram
    participant W as Warp调度器
    participant R as 寄存器
    participant C as CUDA Core
    participant M as 共享内存/L1

    W->>W: 1. 选择一个就绪的 Warp
    W->>R: 2. 读取寄存器中的数据
    W->>C: 3. 发送指令到 CUDA Core
    C->>C: 4. 执行计算
    C->>M: 5. 读写内存（如果需要）
    C->>R: 6. 写回结果
```

---

## 3. CUDA Core vs Tensor Core

### 3.1 CUDA Core

**CUDA Core** 是 GPU 的基本计算单元，执行标量运算。

```mermaid
graph LR
    subgraph CUDA_Core["CUDA Core 功能"]
        A["加法: a + b"]
        B["乘法: a × b"]
        C["乘加: a × b + c"]
        D["逻辑运算"]
    end
```

**特点**：
- 每个 CUDA Core 每个时钟周期执行**一条**浮点运算
- 适合通用计算

### 3.2 Tensor Core

**Tensor Core** 是专门用于矩阵乘法的加速单元。

```mermaid
graph LR
    subgraph 标量运算["普通 CUDA Core"]
        A["一次运算：<br/>a × b + c"]
    end

    subgraph 矩阵运算["Tensor Core"]
        B["一次运算：<br/>4×4 × 4×4 矩阵乘法<br/>= 64 次乘加运算"]
    end

    标量运算 -->|慢| Result["结果"]
    矩阵运算 -->|快很多| Result
```

**对比**：

| 特性 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| **运算类型** | 标量运算 | 矩阵运算 |
| **每次运算量** | 1 次 FP32 运算 | 64 次矩阵运算 |
| **适用场景** | 通用计算 | 深度学习（矩阵乘法） |
| **编程接口** | 普通 C++ 代码 | WMMA API / cuBLAS |

### 3.3 为什么深度学习需要 Tensor Core？

```mermaid
graph TB
    A["神经网络训练"] --> B["核心操作：矩阵乘法"]
    B --> C["例如：GEMM<br/>C = A × B + C"]

    C --> D["A 是 m×k 矩阵<br/>B 是 k×n 矩阵"]
    D --> E["需要 m×k×n 次乘加运算"]

    E --> F["Tensor Core 专为这类运算优化！"]

    style F fill:#90EE90
```

---

## 4. Warp：GPU 的调度单位

### 4.1 什么是 Warp？

**Warp** 是 GPU 线程调度的最小单位，一个 Warp 包含 **32 个线程**。

```mermaid
graph TB
    subgraph Block["一个 Block"]
        subgraph Warp0["Warp 0 (线程 0-31)"]
            W0_T0["T0"] --- W0_T1["T1"] --- W0_T2["..."] --- W0_T31["T31"]
        end
        subgraph Warp1["Warp 1 (线程 32-63)"]
            W1_T0["T32"] --- W1_T1["T33"] --- W1_T2["..."] --- W1_T31["T63"]
        end
        subgraph Warp2["Warp 2 (线程 64-95)"]
            W2_T0["T64"] --- W2_T1["T65"] --- W2_T2["..."] --- W2_T31["T95"]
        end
    end
```

### 4.2 Warp 执行模型

**关键规则**：同一 Warp 中的 32 个线程**同时执行同一条指令**。

```mermaid
sequenceDiagram
    participant W as Warp 调度器
    participant T0 as 线程 0-31

    Note over W: 同一时刻，同一 Warp 的所有线程
    W->>T0: 执行指令 1: ld.global.f32 (加载)
    T0-->>W: 32 个线程同时完成加载
    W->>T0: 执行指令 2: add.f32 (加法)
    T0-->>W: 32 个线程同时完成加法
    W->>T0: 执行指令 3: st.global.f32 (存储)
    T0-->>W: 32 个线程同时完成存储
```

### 4.3 Warp Divergence（分支分歧）

**问题**：当 Warp 内的线程遇到不同的分支时，会发生什么？

```cpp
// 代码示例
if (threadIdx.x < 16) {
    // 分支 A：线程 0-15 执行
    result = a + b;
} else {
    // 分支 B：线程 16-31 执行
    result = a - b;
}
```

```mermaid
graph TB
    subgraph 问题["Warp Divergence 问题"]
        A["Warp 中的 32 个线程"]

        A -->|"线程 0-15"| B["分支 A: 加法"]
        A -->|"线程 16-31"| C["分支 B: 减法"]

        B --> D["第一个周期：<br/>执行分支 A<br/>线程 16-31 等待"]
        C --> E["第二个周期：<br/>执行分支 B<br/>线程 0-15 等待"]

        D --> F["性能下降！<br/>本来可以同时执行的<br/>现在变成了串行"]
        E --> F
    end
```

**解决方案**：尽量让 Warp 内的线程走相同的分支路径。

---

## 5. GPU 内存层级

### 5.1 内存层级图

```mermaid
graph TB
    subgraph 快速内存["快速内存（延迟低）"]
        Reg["寄存器 (Register)<br/>每线程私有<br/>~1 周期<br/>最快"]
        Shared["共享内存 (Shared Memory)<br/>Block 内共享<br/>~20 周期"]
        L1["L1 Cache<br/>每个 SM 一个<br/>~30 周期"]
    end

    subgraph 中速内存["中速内存"]
        L2["L2 Cache<br/>所有 SM 共享<br/>~100 周期"]
    end

    subgraph 慢速内存["慢速内存（带宽大）"]
        Global["全局内存 (Global Memory)<br/>所有线程可见<br/>~400 周期<br/>最大"]
    end

    Reg --> Shared --> L1 --> L2 --> Global

    style Reg fill:#90EE90
    style Shared fill:#98FB98
    style L1 fill:#ADFF2F
    style L2 fill:#FFD700
    style Global fill:#FFA500
```

### 5.2 各级内存详解

| 内存类型 | 位置 | 访问速度 | 可见性 | 大小（典型值） |
|----------|------|----------|--------|----------------|
| **寄存器** | SM 内部 | 1 周期 | 线程私有 | 256 KB / SM |
| **共享内存** | SM 内部 | ~20 周期 | Block 内共享 | 164 KB / SM |
| **L1 Cache** | SM 内部 | ~30 周期 | SM 内共享 | 128 KB / SM |
| **L2 Cache** | GPU 全局 | ~100 周期 | 全局共享 | 40 MB |
| **全局内存** | HBM | ~400 周期 | 全局共享 | 80 GB |

### 5.3 内存访问原则

```mermaid
graph LR
    A["数据访问频率"] -->|"高频"| B["寄存器/共享内存"]
    A -->|"中频"| C["L1/L2 Cache"]
    A -->|"低频"| D["全局内存"]

    style B fill:#90EE90
    style D fill:#FFA500
```

**优化原则**：
1. **高频访问的数据** → 放在寄存器或共享内存
2. **利用数据局部性** → 让 L1/L2 Cache 发挥作用
3. **减少全局内存访问** → 这是性能的主要瓶颈

---

## 6. Block 如何映射到 SM

### 6.1 映射规则

```mermaid
graph TB
    subgraph Grid["Grid (用户定义)"]
        B0["Block 0"]
        B1["Block 1"]
        B2["Block 2"]
        B3["Block 3"]
        B4["Block 4"]
        B5["Block 5"]
    end

    subgraph Hardware["硬件 (SM)"]
        SM0["SM 0"]
        SM1["SM 1"]
        SM2["SM 2"]
    end

    B0 -->|"可能映射到"| SM0
    B1 -->|"可能映射到"| SM1
    B2 -->|"可能映射到"| SM2
    B3 -->|"可能映射到"| SM0
    B4 -->|"可能映射到"| SM1
    B5 -->|"可能映射到"| SM2
```

**关键点**：
1. **Block 到 SM 的映射由硬件自动完成**，程序员无法控制
2. **一个 Block 只能在一个 SM 上执行**
3. **一个 SM 可以同时执行多个 Block**（如果资源足够）

### 6.2 资源限制

一个 SM 能同时运行多少个 Block，取决于：

```mermaid
graph TB
    A["SM 的 Block 容量限制"] --> B["每个 Block 的线程数"]
    A --> C["每个 Block 的共享内存使用量"]
    A --> D["每个 Block 的寄存器使用量"]

    B --> E["最大 2048 线程 / SM<br/>如果一个 Block 有 256 线程<br/>最多 8 个 Block"]
    C --> F["例如：164 KB 共享内存 / SM<br/>如果每个 Block 用 16 KB<br/>最多 10 个 Block"]
    D --> G["例如：65536 寄存器 / SM<br/>如果每个 Block 用 8192 个<br/>最多 8 个 Block"]

    E --> H["最终能同时运行的 Block 数<br/>= min(上述限制)"]
    F --> H
    G --> H
```

---

## 7. 计算能力（Compute Capability）

### 7.1 什么是计算能力？

**Compute Capability**（计算能力）是 NVIDIA 用来描述 GPU 功能集的版本号，格式为 `X.Y`。

### 7.2 各代架构对比

| 架构 | 代表 GPU | 计算能力 | SM 数量 | 特性 |
|------|----------|----------|---------|------|
| **Volta** | V100 | 7.0 | 80 | 第一代 Tensor Core |
| **Turing** | RTX 2080 | 7.5 | 46 | RT Core |
| **Ampere** | A100 | 8.0 | 108 | 第三代 Tensor Core |
| **Ampere** | RTX 3090 | 8.6 | 82 | 消费级旗舰 |
| **Ada Lovelace** | RTX 4090 | 8.9 | 128 | 第四代 Tensor Core |
| **Hopper** | H100 | 9.0 | 132 | FP8 支持 |

### 7.3 查看你的 GPU 计算能力

```bash
# 方法 1：nvidia-smi
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 方法 2：CUDA 程序
# 见后续章节
```

---

## 8. 本章小结

### 8.1 知识图谱

```mermaid
mindmap
  root((GPU 硬件架构))
    SM 流多处理器
      CUDA Core 标量运算
      Tensor Core 矩阵运算
      寄存器/共享内存
      Warp 调度器
    Warp
      32 线程一组
      SIMT 执行
      Branch Divergence 问题
    内存层级
      寄存器 最快
      共享内存 Block 内共享
      L1/L2 Cache
      全局内存 最大
    计算能力
      版本号 X.Y
      功能集标识
```

### 8.2 关键要点

1. **SM 是基本计算单元**：包含多个 CUDA Core、Tensor Core 和内存资源
2. **Warp 是调度单位**：32 个线程同时执行同一条指令
3. **内存层级很重要**：寄存器 > 共享内存 > L1/L2 > 全局内存
4. **Block 映射由硬件决定**：一个 Block 只能在一个 SM 上执行

### 8.3 思考题

1. 如果一个 Block 有 100 个线程，它会被分成几个 Warp？
2. 为什么共享内存比全局内存快这么多？
3. Tensor Core 和 CUDA Core 有什么区别？各适合什么场景？

---

## 下一章

[第四章：线程层级结构](./04_线程层级结构.md) - 深入理解 CUDA 的线程组织方式

---

*参考资料：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | [NVIDIA GPU Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)*