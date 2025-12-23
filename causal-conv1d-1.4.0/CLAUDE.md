[根目录](../CLAUDE.md) > **causal-conv1d-1.4.0**

# causal-conv1d 模块文档

## 模块职责
causal-conv1d 是一个CUDA加速的因果深度卷积1D实现，提供了高效的PyTorch接口。该模块为Mamba模型提供快速的因果卷积操作支持，支持fp32、fp16、bf16数据类型和2、3、4的卷积核大小。

## 入口与启动
模块入口文件为 `causal_conv1d/__init__.py`，主要导出 `causal_conv1d_fn` 和 `causal_conv1d_update` 函数：

- `causal_conv1d_fn`: 基本因果卷积函数
- `causal_conv1d_update`: 卷积状态更新函数

## 对外接口
### causal_conv1d_fn
```
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
```

### causal_conv1d_update
```
def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.

    out: (batch, dim) or (batch, dim, seqlen)
    """
```

## 关键依赖与配置
- PyTorch
- CUDA
- 自定义CUDA内核（causal_conv1d_cuda）

## 数据模型
该模块主要处理以下数据格式：
- 输入张量：(batch, dim, seqlen)
- 卷积权重：(dim, width)
- 输出张量：(batch, dim, seqlen)

## 测试与质量
包含全面的测试套件，位于 `tests/test_causal_conv1d.py`，测试覆盖了：
- 不同数据类型（fp32, fp16, bf16）
- 不同卷积核大小（2, 3, 4）
- 不同序列长度
- 初始状态和最终状态返回
- 偏置和激活函数选项
- 不同维度大小

## 常见问题 (FAQ)
1. **Q: 该模块支持哪些CUDA架构？**
   A: 支持多种CUDA架构，通过CUDA内核实现高效操作

2. **Q: 如何在AMD GPU上使用？**
   A: 对于ROCm 6.0，需要应用提供的补丁文件以避免编译错误

## 相关文件清单
- `causal_conv1d/__init__.py`: 模块入口
- `causal_conv1d/causal_conv1d_interface.py`: 主要接口实现
- `csrc/`: CUDA源代码
- `tests/test_causal_conv1d.py`: 测试代码

## 变更记录 (Changelog)
- 2025-12-23 10:30:42: 初始化模块文档