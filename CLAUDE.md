# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库是一套 **CUDA 入门到中级教学代码 + Agent 互动授课** 代码库，适合配合博客系列（face2ai.com）和互动实验带课使用。用户通常持有 8 卡 GPU 机器，但教学上先从单卡稳定跑通，再扩展到多卡对比。

对应博客系列与课程进度见 `COURSE_OUTLINE_CN.md`，Agent 的互动授课规范见 `AGENTS.md`。

## 构建

```bash
# 完整构建（从仓库根目录）
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"
make -j$(nproc)

# 只编译单个示例（快速迭代）
cmake --build build --target hello_world
cmake --build build --target reduceInteger

# 单独指定目标架构（加快编译）
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3090 / 4090 = 86/89，A100 = 80，H100 = 90
```

快速运行单个章节（无需 CMake）：
```bash
nvcc -I include 3_sum_arrays/sum_arrays.cu -o /tmp/sum_arrays && /tmp/sum_arrays
```

动态并行章节需额外参数：
```bash
nvcc -arch=sm_70 -I include 13_nested_hello_world/nested_Hello_World.cu \
     -o /tmp/nested -lcudadevrt --relocatable-device-code true
```

## 仓库结构

| 目录编号 | 主题 | 关键概念 |
|----------|------|----------|
| 0–2 | Hello World、维度检查、Grid/Block | kernel 启动、线程坐标、向上取整配置 |
| 3–6 | 向量/矩阵加法、计时 | host/device 数据路径、CPU 计时与异步提交 |
| 7–8 | 设备信息、分支分化 | `cudaDeviceProp`、SIMT、warp divergence |
| 9–12 | 矩阵 kernel、归约、循环展开 | 二维索引、reduction、unroll 优化 |
| 13 | 动态并行 | device 端启动子网格，需 `cudadevrt` |
| 14–17 | 全局变量、Pinned/Zero-Copy/UVA 内存 | 各类 host-device 内存管理 API |
| 18–23 | 访问模式、AoS/SoA、统一内存 | 内存对齐、合并访问、`cudaMallocManaged` |
| 24–27 | 共享内存、常量内存、只读缓存 | `__shared__`、bank conflict、广播 |
| 28–29 | Warp Shuffle | `__shfl*_sync`、寄存器间直接通信 |
| 30–38 | CUDA Streams、异步 API、回调 | 多流并发、H2D-kernel-D2H 流水线、callback |

所有示例共用 `include/freshman.h`，提供：`CHECK()`（CUDA API 错误检查）、`cpuSecond()`（CPU 计时）、`initialData()`、`checkResult()`、`initDevice()`、`printMatrix()`。

## 特殊构建情况

- **`13_nested_hello_world`**：CMakeLists 已启用 `CUDA_SEPARABLE_COMPILATION ON` 并链接 `cudadevrt`，编译时需 compute capability ≥ 3.5。
- **`31_stream_omp`**：依赖 OpenMP，未安装时 CMake 会静默跳过该 target。

## 新增示例的步骤

1. 创建 `NN_name/name.cu`，包含标准 include 顺序（`cuda_runtime.h` → `stdio.h` → `freshman.h`）和 `main()`。
2. 创建 `NN_name/CMakeLists.txt`，内容为一行 `add_executable(name name.cu)`。
3. 在根 `CMakeLists.txt` 末尾按数字顺序添加 `add_subdirectory(NN_name)`。
4. CUDA API 调用一律用 `CHECK()` 包裹；`main()` 末尾调用 `cudaDeviceReset()`。
