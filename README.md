# FlashAttention

基于 **Ascend C** 的 FlashAttention（FA）融合算子实现，面向华为昇腾 NPU，用于高效计算 Transformer 中的 Self-Attention。

## 背景

标准 Attention 的计算复杂度为 O(n²)，序列变长时显存与带宽压力显著。FlashAttention 通过**分块计算**与**在线 Softmax**，在不改变数学等价性的前提下，将中间结果尽量保留在片上缓存（UB/L1），减少对全局内存（HBM/GM）的访问。

昇腾 AI Core 采用 **Cube（矩阵）+ Vector（向量）** 协同架构。本项目在 CANN 算子框架下，用 Ascend C 实现 FlashAttention 的前向（及后续计划中的反向）计算逻辑，并针对 Tiling、流水并行与负载均衡做 NPU 侧优化。

## 功能特性

- [ ] FlashAttention 前向（Forward）算子
- [ ] FlashAttention 反向（Backward）算子
- [ ] 支持 Causal Mask（Decoder 场景）
- [ ] 支持变长序列（Varlen）
- [ ] 支持 GQA / MQA（多 Query 头共享 KV 头）
- [ ] PyTorch（`torch_npu`）与 aclnn 单算子调用适配

> 当前仓库处于早期开发阶段，上述能力将随实现进度逐步勾选。

## 环境要求

| 组件 | 说明 |
|------|------|
| 硬件 | Atlas 训练/推理系列（如 Atlas 800T A2、Atlas 800I A2 等） |
| CANN | 建议 ≥ 8.0.RC2，需安装 `ascend-toolkit` |
| 驱动 | 与 CANN 版本匹配的 NPU 驱动与固件 |
| 编译 | CMake ≥ 3.19，支持 C++17 的编译器 |
| Python（可选） | Python 3.8+，`torch` + `torch_npu`（用于框架侧验证） |

安装 CANN 后，请设置环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 项目结构

工程遵循 CANN 自定义算子标准布局（可通过 `msOpGen` 生成骨架后在此基础上开发）：

```
flashattention/
├── build.sh                 # 编译入口脚本
├── cmake/                   # 公共编译脚本
├── CMakeLists.txt
├── CMakePresets.json        # 芯片型号、CANN 路径等编译配置
├── framework/               # AI 框架适配插件（可选）
├── op_host/                 # Host 侧：原型注册、Shape 推导、Tiling
│   ├── flash_attention_tiling.h
│   └── flash_attention.cpp
├── op_kernel/               # Kernel 侧：Ascend C 核函数实现
│   └── flash_attention.cpp
├── scripts/                 # 打包脚本
├── examples/                # 单算子调用示例（计划中）
└── tests/                   # 精度与性能测试（计划中）
```

### 核心模块说明

| 目录/文件 | 职责 |
|-----------|------|
| `op_kernel/` | QK^T、Softmax、PV 等分块计算；Cube/Vector 流水调度 |
| `op_host/` | 输入 Shape 校验、Tiling 参数切分、Workspace 大小计算 |
| `framework/` | 对接 PyTorch / MindSpore 等框架的算子插件 |

## 编译与安装

1. 修改 `CMakePresets.json` 中的关键配置：

   - `ASCEND_CANN_PACKAGE_PATH`：CANN 安装路径，例如 `/usr/local/Ascend/ascend-toolkit/latest`
   - `ASCEND_COMPUTE_UNIT`：目标芯片型号，例如 `ascend910b`

2. 编译算子包：

```bash
./build.sh
```

3. 安装生成的 `.run` 包：

```bash
./build_out/custom_opp_<version>_linux-<arch>.run
```

安装完成后，自定义算子将部署到 CANN 算子库，可通过 aclnn 或框架插件调用。

## 使用方法

### aclnn 单算子调用

CANN 提供 `aclnnFlashAttentionScore` 融合算子接口（本项目的自定义实现注册后，调用方式与之类似）：

```cpp
// 1. 获取 Workspace 大小与执行器
aclnnFlashAttentionScoreGetWorkspaceSize(..., &workspaceSize, &executor);

// 2. 申请 Workspace 并执行
aclnnFlashAttentionScore(workspace, workspaceSize, executor, stream);
```

详细参数与数据排布说明见 [aclnnFlashAttentionScore 文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/appdevgapi/context/FlashAttentionScore.md)。

### PyTorch（torch_npu）

训练场景可通过 `torch_npu.npu_fusion_attention` 调用昇腾 FA 融合算子：

```python
import torch
import torch_npu

output = torch_npu.npu_fusion_attention(
    query, key, value,
    head_num=num_heads,
    input_layout="BNSD",  # 推荐 BNSD 排布以利于连续搬运
    scale=scale,
    keep_prob=1.0 - dropout_p,
    sparse_mode=0,        # Causal 场景参考文档设置 sparse_mode
)
```

接口适配与规格限制可参考 [PyTorch 训练迁移调优指南 - FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)。

## 算法概要

FlashAttention 将 Attention 按块迭代计算，避免物化完整的 S×S Attention 矩阵：

```
对于每个 Q 块：
    初始化在线 Softmax 统计量 (m, l)
    对于每个 K/V 块：
        S_ij = Q_i @ K_j^T * scale
        应用 Mask（可选）
        在线更新 Softmax，累加 PV 贡献
    写出 O_i
```

在昇腾实现中，典型优化方向包括：

- **Tiling**：在 UB 容量约束下尽量增大基本块，减少循环与搬运次数
- **CV 流水**：Cube 做 MatMul，Vector 做 Softmax/Mask，重叠执行
- **核间负载均衡**：Causal Mask 下按块均匀分配 AI Core
- **内存对齐**：Workspace 512B 对齐；优先使用 BNSD 等连续排布

## 输入规格（参考）

以下为 CANN 官方 FlashAttentionScore 的常见约束，自定义实现将尽量对齐：

| 参数 | 说明 |
|------|------|
| 数据类型 | float16、bfloat16（训练时支持梯度） |
| Head 数 | 1 ~ 256 |
| 序列长度 | 1 ~ 512K |
| Layout | BSH、BNSD、SBH、BSND 等（推荐 BNSD） |
| Mask | 可选 Causal / Custom Mask |

具体以算子实现及 Host 侧校验逻辑为准。

## 开发与调试

1. **算子工程生成**（可选，用于初始化骨架）：

```bash
msopgen gen -i flash_attention.json -c ai_core-<soc_version> -lan cpp -out ./FlashAttention
```

2. **精度验证**：使用 CANN `msprof` / 单算子对比工具，与 CPU 参考实现或 CANN 内置 FA 对比

3. **性能分析**：通过 Profiling 查看 Cube、Vector、MTE2、FixPipe 等流水占比，针对性优化

## 参考资料

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [Ascend C 算子开发快速入门](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html)
- [基于 Ascend C 的 FlashAttention 性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240607-1)
- [ops-transformer 算子库](https://gitee.com/ascend/ops-transformer)（昇腾 Transformer 算子开源实现，含 FA 参考）

## 贡献

欢迎提交 Issue 与 Pull Request。提交前请确保：

- 代码符合 Ascend C 编程规范
- 补充必要的精度测试说明
- 重大性能改动附 Profiling 对比数据

## 许可证

许可证待定。若你计划开源，可添加 `LICENSE` 文件并在本节注明协议类型。
