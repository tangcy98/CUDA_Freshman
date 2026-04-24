# CUDA Freshman 中文课程大纲与 Agent 授课指南

本项目适合作为 CUDA 入门到中级性能优化的实验课程。学习方式建议采用“每章阅读代码、编译运行、改一个参数、解释一次结果”的节奏；用户手上有 8 卡 GPU 机器，因此课程中应鼓励观察设备属性、选择不同 GPU、比较单流/多流/多进程场景下的行为，但每个实验先从 `device 0` 稳定跑通。

## 博客与代码导学地图

README 中的博客系列更像理论讲义，本仓库编号目录更像实验讲义。授课时建议先用博客建立概念地图，再用代码验证一个具体机制，最后用 8 卡机器做拓展观察。下面的对应关系可作为 Agent 带课时的主线。

| 博客讲义 | 对应代码 | 教学深化重点 | 建议互动实验 |
| --- | --- | --- | --- |
| [0.0 腾讯云CUDA环境搭建](https://face2ai.com/CUDA-F-0-0-Tencent-GPU-Cloud/) | 全项目构建 | 环境不是课程附属品，而是第一门课；把驱动、CUDA Toolkit、CMake、GPU 可见性分层检查。 | 让用户依次运行 `nvidia-smi`、`nvcc --version`、`cmake --build build -j`，解释每一步验证的是哪一层。 |
| [1.0 并行计算与计算机架构](https://face2ai.com/CUDA-F-1-0-%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97%E4%B8%8E%E8%AE%A1%E7%AE%97%E6%9C%BA%E6%9E%B6%E6%9E%84/) | 课程导入 | 把 CUDA 放在并行计算、计算机架构、吞吐优化的大图里，而不是只当成一种 C 语法扩展。 | 让用户列出一个适合 GPU 的任务和一个不适合 GPU 的任务，并说明并行度与数据搬运成本。 |
| [1.1 异构计算与CUDA](https://face2ai.com/CUDA-F-1-1-%E5%BC%82%E6%9E%84%E8%AE%A1%E7%AE%97-CUDA/) | `0_hello_world` | 建立 host/device 分工：CPU 是调度者，GPU 是吞吐计算设备，二者通过内存和运行时 API 协作。 | 修改 hello world 的 block/thread 数量，观察 device 端输出由多少线程产生。 |
| [2.0 CUDA编程模型概述(一)](https://face2ai.com/CUDA-F-2-0-CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B01/) | `1_check_dimension`、`2_grid_block`、`3_sum_arrays` | 串起 CUDA 程序结构、内存管理、线程管理三件事：先分配和拷贝，再启动 kernel，最后同步和校验。 | 让用户画出 `sum_arrays` 中 host/device 指针流向图。 |
| [2.1 CUDA编程模型概述(二)](https://face2ai.com/CUDA-F-2-1-CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B02/) | `3_sum_arrays`、`include/freshman.h` | 深化 kernel 启动、错误处理、结果验证。重点讲 kernel 启动失败和异步执行错误为什么要显式检查。 | 故意把 block 配置改坏或去掉边界判断，让用户定位错误来源。 |
| [2.2 给核函数计时](https://face2ai.com/CUDA-F-2-2-%E6%A0%B8%E5%87%BD%E6%95%B0%E8%AE%A1%E6%97%B6/) | `4_sum_arrays_timer` | 区分 CPU 计时、CUDA event、profiling 工具；理解异步提交导致的计时误判。 | 让用户分别计 kernel、memcpy、端到端时间，并解释差异。 |
| [2.3 组织并行线程](https://face2ai.com/CUDA-F-2-3-%E7%BB%84%E7%BB%87%E5%B9%B6%E8%A1%8C%E7%BA%BF%E7%A8%8B/) | `5_thread_index`、`6_sum_matrix`、`9_sum_matrix2D`、`11_simple_sum_matrix2D` | 由线程坐标映射到数据坐标；二维矩阵实验用来讲清 block/grid 形状和线性内存布局。 | 改 block 形状，让用户预测 `ix`、`iy`、`idx` 的变化。 |
| [2.4 GPU设备信息](https://face2ai.com/CUDA-F-2-4-%E8%AE%BE%E5%A4%87%E4%BF%A1%E6%81%AF/) | `7_device_information` | 把设备属性和后续优化联系起来：SM 数、warp size、共享内存、最大线程数、copy engine。 | 在 8 卡机器上逐卡查询，比较每张卡是否同型号、同显存、同 compute capability。 |
| [3.1 CUDA执行模型概述](https://face2ai.com/CUDA-F-3-1-CUDA%E6%89%A7%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/) | `6_sum_matrix`、`8_divergence` | 从编程模型进入执行模型：SM、SIMT、warp、占用率、profile-driven optimization。 | 让用户解释为什么“更多线程”不等于“更快”，必须看 SM 资源和访存。 |
| [3.2 线程束执行 Part I](https://face2ai.com/CUDA-F-3-2-%E7%90%86%E8%A7%A3%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%89%A7%E8%A1%8C%E7%9A%84%E6%9C%AC%E8%B4%A8-P1/) | `8_divergence`、`10_reduceInteger` | 线程块是逻辑组织，warp 才是硬件执行基本单位；分支分化必须按 warp 观察。 | 让用户按 `threadIdx.x % warpSize` 推断哪些线程会一起执行。 |
| [3.2 线程束执行 Part II](https://face2ai.com/CUDA-F-3-2-%E7%90%86%E8%A7%A3%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%89%A7%E8%A1%8C%E7%9A%84%E6%9C%AC%E8%B4%A8-P2/) | `6_sum_matrix`、`12_reduce_unrolling` | 用资源分配、延迟隐藏、可扩展性解释性能，而不是只看单个 kernel 的指令数。 | 改 block size，讨论活跃 warp、寄存器、共享内存之间的取舍。 |
| [3.3 并行性表现](https://face2ai.com/CUDA-F-3-3-%E5%B9%B6%E8%A1%8C%E6%80%A7%E8%A1%A8%E7%8E%B0/) | `6_sum_matrix`、`11_simple_sum_matrix2D` | 通过不同线程配置观察活跃 warp 和内存操作，建立 profiling 思维。 | 让用户记录几组 block 配置耗时，解释哪组更接近硬件友好。 |
| [3.4 避免分支分化](https://face2ai.com/CUDA-F-3-4-%E9%81%BF%E5%85%8D%E5%88%86%E6%94%AF%E5%88%86%E5%8C%96/) | `8_divergence`、`10_reduceInteger` | 以归约为例讲分支分化：相邻配对容易造成 warp 内路径不同，交错配对更规整。 | 让用户比较不同 reduction kernel 的耗时和结果。 |
| [3.5 展开循环](https://face2ai.com/CUDA-F-3-5-%E5%B1%95%E5%BC%80%E5%BE%AA%E7%8E%AF/) | `12_reduce_unrolling`、`21_sum_array_offset_unrolling` | 循环展开减少循环控制和分支开销，也会改变寄存器压力；优化不是单调免费的。 | 让用户对比 unroll 2/4/8 的速度，并讨论何时不再变快。 |
| [3.6 动态并行](https://face2ai.com/CUDA-F-3-6-%E5%8A%A8%E6%80%81%E5%B9%B6%E8%A1%8C/) | `13_nested_hello_world` | device 端启动子网格适合动态任务生成，但有编译要求和启动开销。 | 运行嵌套 hello world，画出父网格和子网格的执行关系。 |
| [4.0 全局内存](https://face2ai.com/CUDA-F-4-0-%E5%85%A8%E5%B1%80%E5%86%85%E5%AD%98/) | `15_pine_memory` 到 `23_sum_array_uniform_memory` | 第四章从“怎么算”转向“数据怎么到达计算单元”；全局内存带宽会成为核心瓶颈。 | 让用户先估算一次向量加法的读写字节数，再看实测时间。 |
| [4.1 内存模型概述](https://face2ai.com/CUDA-F-4-1-%E5%86%85%E5%AD%98%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/) | `14_global_variable`、`24_shared_memory_read_data`、`27_stencil_1d_constant_read_only` | 构建 CUDA 内存层次图：寄存器、本地、共享、常量、纹理/只读、全局、统一内存。 | 让用户为每类内存写一句“谁能读写、生命周期、速度特征”。 |
| [4.2 内存管理](https://face2ai.com/CUDA-F-4-2-%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86/) | `15_pine_memory`、`16_zero_copy_memory`、`17_UVA`、`23_sum_array_uniform_memory` | 对比 pageable、pinned、mapped、UVA、managed memory；重点是便利性、可异步性和迁移成本。 | 分别运行 pinned/zero-copy/managed 示例，要求用户解释每种内存在哪里。 |
| [4.3 内存访问模式](https://face2ai.com/CUDA-F-4-3-%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE%E6%A8%A1%E5%BC%8F/) | `18_sum_array_offset`、`19_AoS`、`20_SoA`、`21_sum_array_offset_unrolling` | 对齐、合并访问、缓存加载、AoS/SoA 是全局内存优化的主战场。 | 改 offset 和 block size，记录带宽；让用户用地址连续性解释结果。 |
| [4.4 核函数可达到的带宽](https://face2ai.com/CUDA-F-4-4-%E6%A0%B8%E5%87%BD%E6%95%B0%E5%8F%AF%E8%BE%BE%E5%88%B0%E7%9A%84%E5%B8%A6%E5%AE%BD/) | `22_transform_matrix2D` | 用矩阵转置建立性能上下界：copy 是理想上界，naive transpose 是典型下界。 | 让用户比较 copy、naive、unroll、diagonal 等版本，并画出读写是否合并。 |
| [4.5 使用统一内存的向量加法](https://face2ai.com/CUDA-F-4-5-%E4%BD%BF%E7%94%A8%E7%BB%9F%E4%B8%80%E5%86%85%E5%AD%98%E7%9A%84%E5%90%91%E9%87%8F%E5%8A%A0%E6%B3%95/) | `23_sum_array_uniform_memory` | 统一内存减少显式 `cudaMemcpy`，但并没有消灭数据迁移；同步边界仍然重要。 | 让用户去掉 kernel 后同步，观察 host 访问结果前为什么需要等待。 |
| [5.0 共享内存和常量内存](https://face2ai.com/CUDA-F-5-0-%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E5%92%8C%E5%B8%B8%E9%87%8F%E5%86%85%E5%AD%98/) | `24_shared_memory_read_data` 到 `29_reduce_shfl` | 第五章进入片上数据复用：用共享内存解决全局内存非合并或重复访问问题。 | 让用户先说出一个需要线程块内协作的例子，再看 shared memory 代码。 |
| [5.1 CUDA共享内存概述](https://face2ai.com/CUDA-F-5-1-CUDA%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E6%A6%82%E8%BF%B0/) | `24_shared_memory_read_data` | 共享内存是可编程的片上缓存，作用域是 block，必须用同步保证数据可见。 | 让用户指出 `__shared__` 数据何时写、何时读、为何需要 `__syncthreads()`。 |
| [5.2 共享内存的数据布局](https://face2ai.com/CUDA-F-5-2-%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84%E6%95%B0%E6%8D%AE%E5%B8%83%E5%B1%80/) | `24_shared_memory_read_data`、`26_transform_shared_memory` | 行主序/列主序、静态/动态共享内存、padding 与 bank conflict。 | 让用户把共享内存二维数组映射到 bank，解释 padding 为什么有效。 |
| [5.3 减少全局内存访问](https://face2ai.com/CUDA-F-5-3-%E5%87%8F%E5%B0%91%E5%85%A8%E5%B1%80%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE/) | `25_reduce_integer_shared_memory` | 用共享内存实现 block 内归约，减少全局内存往返。 | 对比 `10_reduceInteger` 和 `25_reduce_integer_shared_memory` 的访存路径。 |
| [5.4 合并的全局内存访问](https://face2ai.com/CUDA-F-5-4-%E5%90%88%E5%B9%B6%E7%9A%84%E5%85%A8%E5%B1%80%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE/) | `26_transform_shared_memory` | 用共享内存 tile 改造矩阵转置，使全局读写都更接近合并访问。 | 让用户比较无 shared 和 shared transpose，解释 tile 在哪里发生转置。 |
| [5.5 常量内存](https://face2ai.com/CUDA-F-5-5-%E5%B8%B8%E9%87%8F%E5%86%85%E5%AD%98/) | `27_stencil_1d_constant_read_only` | 常量内存适合小规模只读参数，warp 内同址读取可广播；只读缓存适合另一类读取模式。 | 修改 stencil 半径或系数读取方式，讨论是否仍适合常量内存。 |
| [5.6 线程束洗牌指令](https://face2ai.com/CUDA-F-5-6-%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%B4%97%E7%89%8C%E6%8C%87%E4%BB%A4/) | `28_shfl_test`、`29_reduce_shfl` | shuffle 让 warp 内线程直接交换寄存器数据，减少共享内存和同步开销；现代 CUDA 使用 `_sync` API。 | 逐个运行 broadcast/up/down/xor，再让用户画出 lane 数据流。 |
| [6.0 流和并发](https://face2ai.com/CUDA-F-6-0-%E6%B5%81%E5%92%8C%E5%B9%B6%E5%8F%91/) | `30_stream` 到 `38_stream_call_back` | 第六章从 kernel 内并行上升到应用级并行：多个 kernel、拷贝和 CPU 工作如何重叠。 | 让用户先用一张时间线画出单流串行执行，再引入多流。 |
| [6.1 流和事件概述](https://face2ai.com/CUDA-F-6-1-%E6%B5%81%E5%92%8C%E4%BA%8B%E4%BB%B6%E6%A6%82%E8%BF%B0/) | `30_stream`、`37_asyncAPI` | stream 是有序队列，event 是时间点和依赖标记；同一 stream 保序，不同 stream 可并发。 | 用 event 计时，并让用户解释 event 记录在默认流和非默认流中的含义。 |
| [6.2 并发内核执行](https://face2ai.com/CUDA-F-6-2-%E5%B9%B6%E5%8F%91%E5%86%85%E6%A0%B8%E6%89%A7%E8%A1%8C/) | `30_stream`、`31_stream_omp`、`32_stream_resource`、`33_stream_block`、`34_stream_dependence` | 多 stream 不等于必然并发；资源占用、默认流阻塞、硬件工作队列、stream 间依赖都会影响结果。 | 让用户调大 kernel 工作量，观察并发是否消失，并解释资源限制。 |
| [6.3 重叠内核执行和数据传输](https://face2ai.com/CUDA-F-6-3-%E9%87%8D%E5%8F%A0%E5%86%85%E6%A0%B8%E6%89%A7%E8%A1%8C%E5%92%8C%E6%95%B0%E6%8D%AE%E4%BC%A0%E8%BE%93/) | `35_multi_add_depth`、`36_multi_add_breadth` | depth-first 与 breadth-first 提交顺序影响 H2D、kernel、D2H 三阶段流水线。 | 改 `N_SEGMENT`，让用户画出每个 stream 的三段流水线。 |
| [6.4 重叠GPU和CPU的执行](https://face2ai.com/CUDA-F-6-4-%E9%87%8D%E5%8F%A0GPU%E5%92%8CCPU%E7%9A%84%E6%89%A7%E8%A1%8C/) | `37_asyncAPI` | GPU work 异步排队后，CPU 可以继续执行；`cudaEventQuery` 可用于非阻塞轮询。 | 观察 `cpu counter`，解释它为什么能增长。 |
| [6.5 流回调](https://face2ai.com/CUDA-F-6-5-%E6%B5%81%E5%9B%9E%E8%B0%83/) | `38_stream_call_back` | callback 是把 host 函数排进 stream 的完成通知机制；回调中不能随意调用 CUDA API。 | 让用户预测 callback 打印顺序是否等于 stream 创建顺序，并解释异步完成顺序。 |

授课时不要求用户一次读完所有博客。更好的节奏是：每进入一个主题簇，先读 1 篇总览，再跑 2-4 个代码实验，最后回到博客中的硬件解释。比如执行模型主题读 3.1/3.2，再跑 `8_divergence` 和 `12_reduce_unrolling`；内存主题读 4.1/4.3，再跑 `18_sum_array_offset`、`19_AoS`、`20_SoA`；共享内存主题读 5.1/5.2，再跑 `24_shared_memory_read_data` 和 `26_transform_shared_memory`。

## 课前准备

目标：确认 CUDA 工具链、GPU、驱动和构建方式可用。

建议命令：

```bash
nvidia-smi
nvcc --version
cmake -S . -B build
cmake --build build -j
```

如果目标机器的 GPU 架构明确，也可以指定架构，例如 A100 使用 `-DCMAKE_CUDA_ARCHITECTURES=80`，L40/L4/4090 使用 `89`，H100/H200 使用 `90`。如果全量构建太慢，先进入单章目录用 `nvcc` 或构建对应 target。

授课指导：先让用户读懂 `include/freshman.h` 中的 `CHECK`、`cpuSecond`、初始化和校验函数。强调 CUDA 学习不是先背 API，而是先建立“host 发起、device 执行、显式同步、显式搬运”的执行模型。8 卡机器上第一步只选一张卡，避免把多 GPU 与单 GPU 基础概念混在一起。

## 第 0 章：CUDA Hello World

代码：`0_hello_world/hello_world.cu`

学习目标：理解 `.cu` 文件、host 代码、kernel 启动语法 `<<<grid, block>>>`、`cudaDeviceReset`。

授课指导：让用户先运行程序，再指出 CPU 打印和 GPU 打印的执行顺序可能受同步影响。引导用户改 block/thread 数量，观察输出数量如何变化。不要急着讲性能，先让用户形成“kernel 是大量线程并行执行的函数”的直觉。

## 第 1 章：线程块与网格维度

代码：`1_check_dimension/check_dimension.cu`

学习目标：认识 `dim3`、`gridDim`、`blockDim`、`blockIdx`、`threadIdx`。

授课指导：让用户画出 grid 与 block 的二维结构，再对照程序输出。重点提问：一个线程如何知道自己在整个网格中的位置？如果 block 形状改变，总线程数和索引映射是否改变？

## 第 2 章：Grid/Block 配置计算

代码：`2_grid_block/grid_block.cu`

学习目标：掌握 `(nElem - 1) / block.x + 1` 这种向上取整的 launch 配置。

授课指导：让用户把 `nElem` 改成不能被 block 整除的数，解释为什么 kernel 内部还需要边界判断。这里要建立一个核心习惯：launch 配置负责覆盖数据，kernel 内判断负责不越界。

## 第 3 章：向量加法

代码：`3_sum_arrays/sum_arrays.cu`

学习目标：完成第一条完整 CUDA 数据路径：host 分配、device 分配、H2D 拷贝、kernel、D2H 拷贝、结果校验。

授课指导：逐行追踪 `a_h -> a_d -> res_d -> res_from_gpu_h`。让用户先口述每个指针位于 host 还是 device，再运行验证。随后要求用户给 kernel 增加 `N` 参数和边界判断，理解为什么真实项目不能依赖“数据刚好整除”。

## 第 4 章：Kernel 计时

代码：`4_sum_arrays_timer/sum_arrays_timer.cu`

学习目标：认识 CPU 计时与 CUDA 异步执行之间的关系。

授课指导：让用户对比只包 kernel、包 memcpy+kernel、加入/去掉同步时的时间差。强调“host 计时看到的是提交工作还是完成工作”这个问题，为后续 stream 和 event 铺垫。

## 第 5 章：线程索引

代码：`5_thread_index/thread_index.cu`

学习目标：从一维/二维线程索引映射到线性内存。

授课指导：让用户手工计算几个线程访问的数组下标，再运行验证。提问重点是行主序布局、`ix + iy * nx` 的来源，以及错误索引如何导致越界或覆盖。

## 第 6 章：矩阵加法与不同 block 形状

代码：`6_sum_matrix/sum_matrix.cu`

学习目标：比较不同 block 形状对矩阵访问和性能的影响。

授课指导：让用户固定矩阵大小，运行不同配置，记录时间。引导用户从“总线程数一样”过渡到“访存合并、warp 排布、边界线程”这些更细的因素。

## 第 7 章：设备信息查询

代码：`7_device_information/device_information.cu`

学习目标：读懂 GPU 计算能力、SM 数、warp 大小、全局内存、共享内存、最大线程数等属性。

授课指导：在 8 卡机器上让用户逐卡查询并比较。重点不是背参数，而是把参数和后续实验联系起来：最大 block 线程数约束 launch，共享内存大小约束 tile，copy/compute overlap 依赖设备能力。

## 第 8 章：分支分化

代码：`8_divergence/divergence.cu`

学习目标：理解 warp 内 SIMT 执行和 branch divergence。

授课指导：让用户对比相邻线程分支与按 warp 分组分支的耗时。强调分支本身不是罪魁祸首，warp 内线程走不同路径才会序列化。让用户用 `threadIdx.x / warpSize` 改写条件来观察变化。

## 第 9-11 章：二维矩阵 Kernel 与坐标映射

代码：`9_sum_matrix2D/sum_matrix2D.cu`、`10_reduceInteger/reduceInteger.cu`、`11_simple_sum_matrix2D/simple_sum_matrix.cu`

学习目标：巩固二维数据布局，开始接触归约。

授课指导：第 9 和第 11 章用矩阵加法强化二维索引，第 10 章引入 reduction 的“多线程合并为少量结果”。教学时要把“每个线程独立写一个元素”和“多个线程协作写一个结果”分开讲清楚。

## 第 12 章：归约与循环展开

代码：`12_reduce_unrolling/reduceUnrolling.cu`

学习目标：理解 reduce 的多种优化：相邻配对、交错配对、unroll、warp 内展开。

授课指导：让用户每次只运行一个 kernel，记录耗时和结果。引导用户关注三件事：全局内存访问次数、同步次数、warp divergence。最后要求用户解释为什么同一个数学求和会有不同性能。

## 第 13 章：动态并行

代码：`13_nested_hello_world/nested_Hello_World.cu`

学习目标：理解 device 端启动 kernel 的动态并行模型。

授课指导：先说明这不是日常入门代码的默认选择，它有额外编译要求和启动开销。让用户观察递归 depth 输出，再讨论什么场景可能需要 GPU 自己派发后续工作。构建时注意需要 separable compilation 和 `cudadevrt`。

## 第 14 章：全局变量与常量符号

代码：`14_global_variable/global_variable.cu`

学习目标：认识 device/global symbol、host 与 device 间符号拷贝。

授课指导：让用户区分普通 host 全局变量与 device 全局变量。用 `cudaMemcpyToSymbol`/`cudaMemcpyFromSymbol` 解释“名字相同不等于地址空间相同”。

## 第 15-17 章：页锁定内存、Zero Copy 与 UVA

代码：`15_pine_memory/pine_memory.cu`、`16_zero_copy_memory/zero_copy_memory.cu`、`17_UVA/UVA.cu`

学习目标：理解 pageable memory、pinned memory、mapped host memory、Unified Virtual Addressing。

授课指导：先讲 pinned memory 为什么能提升异步拷贝，再讲 zero-copy 是 GPU 直接访问 host 内存，不是免费加速。8 卡机器上要提醒用户 NUMA/PCIe 拓扑会影响结果。让用户比较 zero-copy 与普通 device memory 的耗时，并解释为什么大吞吐计算通常仍应把数据放到 device memory。

## 第 18-21 章：全局内存访问模式、AoS 与 SoA

代码：`18_sum_array_offset/sum_array_offset.cu`、`19_AoS/AoS.cu`、`20_SoA/SoA.cu`、`21_sum_array_offset_unrolling/sum_array_offset_unrolling.cu`

学习目标：理解内存对齐、offset、stride、coalescing、结构体数组和数组结构体对吞吐的影响。

授课指导：让用户多次运行 offset/stride 实验，记录带宽变化。AoS/SoA 章节要结合真实场景讲：如果 kernel 只读取某几个字段，SoA 往往更利于合并访问；如果总是整对象处理，AoS 可能更自然。不要把某种布局讲成绝对正确。

## 第 22-23 章：矩阵转置与统一内存

代码：`22_transform_matrix2D/transform_matrix2D.cu`、`23_sum_array_uniform_memory/sum_arrays_uniform_memory.cu`

学习目标：理解矩阵转置中的非合并访问，以及 Unified Memory 的编程便利性和同步要求。

授课指导：转置实验要让用户观察 naive、按行/列访问、不同 block 形状带来的差别。统一内存章节要强调它减少显式拷贝代码，但不等于没有数据迁移成本；kernel 后访问结果前必须同步。

## 第 24-26 章：共享内存

代码：`24_shared_memory_read_data/shared_memory_read_data.cu`、`25_reduce_integer_shared_memory/reduce_integer_shared_memory.cu`、`26_transform_shared_memory/transform_shared_memory.cu`

学习目标：掌握 `__shared__`、线程块内协作、`__syncthreads()`、共享内存 bank conflict、tile 转置。

授课指导：先用读写共享内存的小例子解释生命周期和可见范围，再进入 reduction 与 transpose。让用户明确共享内存是 block 内私有缓存，不是所有线程共享的全局缓存。讲 bank conflict 时用地址到 bank 的映射做小表格。

## 第 27 章：常量内存与只读数据

代码：`27_stencil_1d_constant_read_only/stencil_1d_constant_read_only.cu`

学习目标：理解 stencil 模式、常量内存广播、只读缓存。

授课指导：让用户观察同一组系数被大量线程重复读取的模式。说明常量内存适合“小、只读、warp 内访问相同地址”的数据，不适合大数组随机访问。

## 第 28-29 章：Warp Shuffle 与 Shuffle Reduction

代码：`28_shfl_test/shfl_test.cu`、`29_reduce_shfl/reduce_shfl.cu`

学习目标：掌握 warp 内寄存器交换，减少共享内存和同步开销。

授课指导：先让用户分别运行 broadcast/up/down/xor，再画出 lane 之间的数据流。进入 reduction 时强调 shuffle 只在 warp 内通信，跨 warp 仍需要共享内存或其他机制汇总。现代 CUDA 使用 `__shfl*_sync`，要解释 mask 的意义。

## 第 30-31 章：Stream 与 OpenMP 派发

代码：`30_stream/stream.cu`、`31_stream_omp/stream_omp.cu`

学习目标：理解 CUDA stream 的顺序性、不同 stream 的潜在并发，以及 CPU 多线程提交 GPU work。

授课指导：先运行单线程多 stream，再用 OpenMP 多 host 线程提交。提醒用户多 stream 不保证一定并发，是否并发取决于 kernel 资源占用、设备能力和默认 stream 语义。8 卡机器上可以扩展讨论“多 CPU 线程分别绑定不同 GPU”的模式，但不在本章强行实现。

## 第 32-34 章：Stream 资源、阻塞与依赖

代码：`32_stream_resource/stream_resource.cu`、`33_stream_block/stream_block.cu`、`34_stream_dependence/stream_dependence.cu`

学习目标：理解 stream 并发受资源限制，掌握 event/依赖对执行顺序的影响。

授课指导：让用户比较多个小 kernel 与少数大 kernel 的并发情况。讲清楚“stream 是队列，不是 GPU 核心”；当单个 kernel 已占满 SM 或寄存器/共享内存，增加 stream 不会自动变快。

## 第 35-38 章：数据分段、异步拷贝与回调

代码：`35_multi_add_depth/multi_add_depth.cu`、`36_multi_add_breadth/multi_add_breadth.cu`、`37_asyncAPI/asyncAPI.cu`、`38_stream_call_back/stream_call_back.cu`

学习目标：掌握 H2D、kernel、D2H 的流水化；理解 depth-first 与 breadth-first 提交顺序；观察异步 API 和 stream callback。

授课指导：用时间线画出每个 segment 在每个 stream 里的 H2D/kernel/D2H。让用户分别调大 `N_SEGMENT`、`N_REPEAT`，观察什么时候 overlap 明显。强调异步拷贝需要 pinned host memory 才有意义；callback 示例主要用于理解完成通知，现代项目也可考虑 `cudaLaunchHostFunc`。

## 课程总复习

建议让用户完成三个小任务：

1. 给 `3_sum_arrays` 增加可配置数据规模和边界判断。
2. 在 `12_reduce_unrolling` 中选两个 kernel，用自己的话解释性能差异。
3. 修改 `35_multi_add_depth` 或 `36_multi_add_breadth` 的 `N_SEGMENT`，画出时间线并解释结果。

评价标准：能正确编译运行、能判断数据是否正确、能解释一次性能变化的主要原因、能说清 host/device 内存和同步边界。

## Agent 授课指南

Agent 的角色不是替用户读完所有代码，而是作为一名互动式实验教练，按章节引导用户“预测、执行、观察、解释、改动、复盘”。每一轮对话都应该围绕一个小目标，避免一次性倾倒大量概念。

### 持久化课程记录管理

课程进度和互动内容统一记录在 `course_records/`。这个目录把信息分成四类：`learner_profile.yaml` 保存长期稳定的学习者画像和硬件环境，`progress.yaml` 保存当前进度和章节状态，`session_log.md` 追加每次互动会话，`chapters/*.md` 保存章节复盘。Agent 每次正式授课都应把这里当成课程状态的事实来源。

Agent 开始一轮教学前，应先读取 `course_records/progress.yaml` 和相关章节笔记，确认当前章节、掌握程度、阻塞点和下一步建议。如果用户提供了新的 GPU 型号、CUDA 版本、学习目标或时间安排，再同步更新 `course_records/learner_profile.yaml`。

每完成一次“预测 -> 执行 -> 观察 -> 解释 -> 改动 -> 复盘”的小循环，Agent 应更新三处记录：在 `progress.yaml` 中更新当前章节状态、掌握程度、阻塞点和下一步；在 `session_log.md` 中追加一条可复盘日志；如果本轮产生了章节级结论，则更新对应的 `chapters/*.md`。记录要短而准，只保存关键命令、关键输出、用户解释和后续动作，不复制大段终端输出。

章节状态只使用 `not_started`、`in_progress`、`blocked`、`review_needed`、`completed`。掌握程度使用 0-4：0 表示未接触，1 表示能跟跑，2 表示能解释主要 API 和数据流，3 表示能解释性能现象并做小改动，4 表示能迁移到新问题或独立调试。Agent 不应因为“程序跑通”就直接标记 completed，必须同时看到用户能解释关键机制。

如果一次实验失败，Agent 要保留失败记录并把章节状态设为 `blocked` 或 `review_needed`，写明下一步排查方向。不要把失败日志覆盖成成功日志；失败是课程证据的一部分。若后续修复成功，再追加新的日志并更新状态。

### 博客与代码的结合方式

每章教学按“博客概念 -> 代码证据 -> 机器观察 -> 用户复述”四步推进。博客负责回答“为什么这个现象重要”，代码负责回答“这个现象在哪里发生”，8 卡机器负责回答“我的硬件上是否也这样”。Agent 不应把博客内容整段复述给用户，而应提炼成 2-3 个问题，引导用户在代码中找到证据。

对于基础章节，先读博客再跑代码。例如学习 `3_sum_arrays` 前，让用户先抓住 2.0/2.1 博客中的 CUDA 程序结构：分配内存、传输数据、启动 kernel、验证结果。跑完代码后追问：“这里哪一步发生在 host，哪一步发生在 device？”

对于性能章节，先跑代码再回到博客。例如学习 `8_divergence`、`12_reduce_unrolling`、`18_sum_array_offset` 时，先让用户观察耗时差异，再回到博客里的 warp、分支分化、合并访问解释。这样用户会带着问题读理论，不会把硬件概念当成孤立术语。

对于内存和并发章节，要求用户画图。内存章节画 host/device/global/shared/constant 的数据流和生命周期；stream 章节画 H2D、kernel、D2H、CPU work 的时间线。Agent 要不断追问“这条线上的依赖是什么”“这两个操作能否重叠”“如果换成 8 卡机器，瓶颈会转移到哪里”。

博客中部分工具或架构背景来自较早 CUDA 时代，例如 `nvprof`、NVVP、旧版 shuffle API、`sm_35`。Agent 应明确区分历史讲义和现代实践：保留博客中的执行模型与优化思想，但在命令和代码上使用本项目已更新的 CMake 架构设置、`__shfl*_sync`、`nvidia-smi`，以及现代 Nsight Systems / Nsight Compute 作为 profiling 延伸方向。

### 开始一章时

先确认用户当前章节、GPU 型号、是否已完成构建。然后给出本章的三个要点：本章要观察什么、要运行哪个 target、运行后应该看哪几行输出。示例开场：

```text
这一章我们只抓一个问题：线程如何把自己映射到数组下标。
先运行 ./build/5_thread_index/thread_index，然后把输出里的 blockIdx/threadIdx 和代码中的 ix/iy 对上。
运行完把输出贴给我，我们一起解释其中两个线程的访问位置。
```

### 每章的互动节奏

1. 预测：先问用户“你觉得输出/耗时会怎样变化”，让用户带着假设运行。
2. 执行：给出最小命令，不一次给太多选项。
3. 观察：让用户贴关键输出，Agent 帮忙筛掉无关噪声。
4. 解释：把输出连接回本章核心概念。
5. 改动：只改一个变量或一小段代码，例如 block size、矩阵维度、`N_SEGMENT`。
6. 复盘：要求用户用一两句话总结现象背后的机制。

### 命令指导原则

优先使用项目构建产物：

```bash
cmake --build build --target sum_arrays
./build/3_sum_arrays/sum_arrays
```

如果用户只想快速跑单章，可以使用 `nvcc`：

```bash
nvcc -I include 3_sum_arrays/sum_arrays.cu -o /tmp/sum_arrays
/tmp/sum_arrays
```

遇到动态并行章节，应提醒需要 relocatable device code：

```bash
nvcc -arch=sm_70 13_nested_hello_world/nested_Hello_World.cu -o /tmp/nested_Hello_World -lcudadevrt --relocatable-device-code true
```

### 8 卡机器上的引导方式

默认先使用 `cudaSetDevice(0)` 或 `CUDA_VISIBLE_DEVICES=0` 跑通单卡。设备信息章节再扩展到 8 卡比较：

```bash
nvidia-smi topo -m
CUDA_VISIBLE_DEVICES=0 ./build/7_device_information/device_information
CUDA_VISIBLE_DEVICES=1 ./build/7_device_information/device_information
```

不要在用户还没掌握 stream、memory copy、synchronization 前引入多 GPU 编程。等用户完成 stream 章节后，再讨论多 GPU 下 host 线程、进程、上下文和 peer access。

### 常见卡点与提示

如果结果不匹配，先查四件事：kernel 是否越界、grid 是否覆盖全量数据、是否漏了同步、是否复制了正确的 device buffer。

如果计时不稳定，先固定 GPU、预热一次、增加数据规模或重复次数，再区分 CPU wall time 与 CUDA event time。

如果多 stream 没有变快，提醒用户检查 pinned memory、kernel 资源占用、拷贝方向、设备是否支持 concurrent copy/execute。

如果编译失败且提示架构不支持，指导用户根据 `nvidia-smi` 和 GPU 型号设置 `CMAKE_CUDA_ARCHITECTURES`，不要继续使用旧的 `sm_35`。

### Agent 的提问模板

每章最多同时问一个问题，问题要能推动用户执行或解释：

```text
你先猜一下：把 block 从 256 改成 1024，结果会变吗？耗时可能怎么变？
```

```text
这次输出里哪一行说明 kernel 已经完成？如果没有同步，host 端会不会提前继续执行？
```

```text
你能用“访存是否合并”解释 AoS 和 SoA 的差别吗？先不用追求术语，按线程访问地址说。
```

### Agent 的纠错方式

发现用户理解偏差时，先肯定其可用部分，再指出边界。例如：“你说统一内存不用手动 `cudaMemcpy` 是对的；但它仍然会发生页迁移，所以性能上不是免费。”避免只给结论，要把结论落到本章代码中的某一行。

### 每章结束标准

一章完成时，Agent 应确认用户做到三件事：能独立运行本章程序，能指出代码中最关键的 CUDA API 或 kernel，能用自己的话解释一个观察到的现象。满足后再进入下一章；如果用户只是跑通但说不清，应追加一个小改动实验，而不是直接推进。
