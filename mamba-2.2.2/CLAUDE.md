[根目录](../CLAUDE.md) > **mamba-2.2.2**

# mamba 模块文档

## 模块职责
mamba 模块是 Mamba 状态空间模型的核心实现，包含 Mamba1 和 Mamba2 架构。该模块基于选择性状态空间（Selective State Spaces）实现线性时间序列建模，特别适用于处理信息密集型数据，如语言建模。

## 入口与启动
模块入口文件为 `mamba_ssm/__init__.py`，主要导出以下核心组件：
- `selective_scan_fn` 和 `mamba_inner_fn`: 底层操作函数
- `Mamba`: Mamba1架构实现
- `Mamba2`: Mamba2架构实现
- `MambaLMHeadModel`: 语言模型头封装

## 对外接口
### Mamba 类
```
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,          # 模型维度
        d_state=16,       # SSM状态扩展因子
        d_conv=4,         # 局部卷积宽度
        expand=2,         # 块扩展因子
        dt_rank="auto",   # 时间步秩
        dt_min=0.001,     # 最小时间步
        dt_max=0.1,       # 最大时间步
        ...
    ):
```

### Mamba2 类
Mamba2 提供了改进的状态空间模型架构，具有更好的性能。

### MambaLMHeadModel 类
完整的语言模型，包含 Mamba 背骨和语言模型头。

## 关键依赖与配置
- PyTorch 1.12+
- CUDA 11.6+
- causal-conv1d>=1.4.0
- einops
- Triton（用于优化内核）

## 数据模型
- 输入张量：(batch, seqlen, dim)
- 输出张量：(batch, seqlen, dim) - 与输入同形状
- 状态张量：用于序列生成和推理缓存

## 测试与质量
- 提供基准测试脚本 `benchmarks/benchmark_generation_mamba_simple.py`
- 包含零样本评估工具和 lm-evaluation-harness 集成
- 支持多种预训练模型的加载和推理

## 常见问题 (FAQ)
1. **Q: Mamba 模型的序列处理复杂度是多少？**
   A: 线性时间复杂度 O(L)，其中 L 是序列长度，这使得它在长序列处理上优于 Transformer 的 O(L²)

2. **Q: Mamba 与 Transformer 的主要区别是什么？**
   A: Mamba 基于选择性状态空间模型，而 Transformer 基于自注意力机制。Mamba 在信息密集型任务上表现更优

3. **Q: 如何使用预训练模型？**
   A: 可以通过 Hugging Face Hub 自动加载 state-spaces 的预训练模型

## 相关文件清单
- `mamba_ssm/__init__.py`: 模块入口
- `mamba_ssm/modules/mamba_simple.py`: Mamba1实现
- `mamba_ssm/modules/mamba2.py`: Mamba2实现
- `mamba_ssm/ops/selective_scan_interface.py`: 选择性扫描操作
- `mamba_ssm/models/mixer_seq_simple.py`: 语言模型封装
- `mamba_ssm/ops/triton/`: Triton优化内核

## 变更记录 (Changelog)
- 2025-12-23 10:30:42: 初始化模块文档