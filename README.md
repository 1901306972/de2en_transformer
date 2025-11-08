# 从零开始实现 Transformer：一个用于德英翻译的模型

## 目录
- [1. 项目简介](#1-项目简介)
- [2. 代码库结构](#2-代码库结构)
- [3. 复现指南](#3-复现指南)
  - [3.1. 环境设置](#31-环境设置)
  - [3.2. 数据准备](#32-数据准备)
  - [3.3. 训练](#33-训练)
  - [3.4. 评估](#34-评估)
  - [3.5. 推理](#35-推理)
- [4. 实验环境与预期结果](#4-实验环境与预期结果)
- [5. 核心发现与总结](#5-核心发现与总结)

## 1. 项目简介

本项目是“大型模型基础与应用”课程的期中作业。其核心目标是**从零开始、仅使用 PyTorch 基础模块，手工搭建一个完整的 Encoder-Decoder 架构的 Transformer 模型**，并在 IWSLT2017 (DE-EN) 数据集上进行训练和评估，以实现一个基础的德语到英语的翻译功能。

通过这个项目，我们深入实践了 Transformer 的所有核心组件，包括：
- 多头自注意力 (Multi-Head Self-Attention)
- 交叉注意力 (Cross-Attention)
- 位置前馈网络 (Position-wise Feed-Forward Network)
- 正弦/余弦位置编码 (Sinusoidal Positional Encoding)
- 残差连接与 Pre-Layer Normalization
- 掩码机制 (Padding & Subsequent Masking)

此外，我们还实现了一套完整的训练与评估流程，集成了学习率调度、标签平滑、早停等现代训练技巧。

## 2. 代码库结构

本仓库遵循了清晰、模块化的项目结构：
```
├── configs/
│   ├── simple.yaml          # "返璞归真"版最终训练配置
│   └── ...                  # 其他实验配置文件
├── data/
│   └── iwslt2017-en-de/    # 本地数据集缓存目录
├── models/                  # 保存训练好的模型权重和词典
├── results/                 # 保存损失曲线图等结果
├── src/
│   ├── init.py
│   ├── data.py              # 数据加载与预处理模块
│   └── model.py             # Transformer 模型定义
├── train.py                 # 主训练脚本
├── evaluate.py              # 最终评估脚本
├── inference.py             # 交互式翻译脚本
├── README.md                # 本文档
└── requirements.txt         # Python 依赖列表
```


## 3. 复现指南

### 3.1. 环境设置

本项目依赖于 Python 3.10 以及一系列科学计算库。我们强烈推荐使用 Conda 来创建和管理虚拟环境。

```bash
# 1. 克隆本仓库
git clone https://github.com/1901306972/de2en_transformer.git
cd de2en_transformer

# 2. 创建并激活 Conda 环境
conda create -n transformer python=3.10 -y
conda activate transformer

# 3. 安装所有 Python 依赖
pip install -r requirements.txt

# 4. 下载 Spacy 所需的语言模型
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

```

### 3.2. 数据准备
本项目使用的 IWSLT2017 (DE-EN) 数据集将由 src/data.py 脚本在首次运行时自动从 datasets 库下载并缓存到本地的 data/ 目录中。您无需手动下载。

### 3.3. 训练
我们所有的实验都通过 train.py 脚本启动，并通过 --config 参数指定配置文件。要复现报告中的最终基线模型，请使用 configs/simple.yaml 在完整数据集上进行训练。

精确的复现命令如下：
```bash
# 假设使用 0 号 GPU 进行单卡训练
# --seed 42 保证了权重初始化和数据处理的随机性可复现

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/simple.yaml

```
训练过程会自动保存验证集上表现最好的模型权重到 models/best_model.pt，并将对应的词典保存为 models/src_vocab.pt 和 models/tgt_vocab.pt。

### 3.4. 评估
训练完成后，可以运行评估脚本来复现模型在测试集上的最终性能指标（Loss, PPL, BLEU）。
```bash
python evaluate.py
```
### 3.5. 推理
我们提供了一个交互式的推理脚本，您可以输入德语句子来直观地检验模型的翻译效果。
```bash
python inference.py
```
## 4. 实验环境与预期结果
硬件环境:

GPU: NVIDIA GeForce RTX 3090 (24GB 显存)
CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
内存 (RAM): 251 GB
预期运行时间:

在完整的 IWSLT2017 数据集上进行训练，每个轮次（epoch）的平均耗时约为 15 分钟。
考虑到早停机制，完整的训练过程预计在 3–4 小时内完成。
预期最终性能:

遵循上述指令，您应该能复现出与我们报告中相似的结果，即在 IWSLT2017 (DE-EN) 测试集上达到约 9–10 的 BLEU-4 分数。
## 5. 核心发现与总结
通过本次从零实现的旅程，我们不仅成功地构建并训练了一个功能正确的 Transformer 模型，更重要的是，在漫长而曲折的调试过程中，我们深刻理解了：

掩码机制对于模型正确性的决定性作用。
训练策略（如学习率调度、正则化）对模型最终性能的巨大影响。
在有限资源下，模型不可避免地会遇到过拟合问题，以及如何通过实验来诊断它。
手工实现与理论之间的差距，以及代码细节（如 Pre-LN 架构、张量连续性）对数值稳定性的关键意义。
最终，我们得到了一个健康的、可工作的基线模型，并为未来如何进一步提升其性能（如扩大模型容量、实现 Beam Search）指明了清晰的方向。
