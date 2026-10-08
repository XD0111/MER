# MER: Multi-Expert Routing for Event Graph Completion

English is expanded by default. Collapse English and expand 中文 to read the Chinese version on this page.

英文默认展开。阅读中文时，收起 English，再展开“中文”。

<details open>
<summary><strong>English</strong></summary>

Official code repository for *Multi-Expert Routing for Event Graph Completion* (EMNLP 2026).

MER (Multi-Expert Routing) is a multi-expert framework for **heterogeneous event graph completion** (EGC). It converts heterogeneous event graphs into structured triple sequences and uses prompt learning with pretrained language models to encode semantic and structural information. **AER** (Adaptive Expert Routing) dynamically selects and aggregates the predictions of five experts for each query event pair, jointly completing four types of event relations:

- **Sub-event**
- **Causality**
- **Temporal**
- **Coreference**

The five experts, named as in Section 2.2 of the paper, are **Mention Expert, Sentence Expert, Context Expert, Path Expert, and Graph Expert**.

The dataset is **EGC-MAVEN**, introduced in our previous work, PLAF (*Neural Networks*, 2026,
[10.1016/j.neunet.2026.108730](https://doi.org/10.1016/j.neunet.2026.108730)),
and constructed from MAVEN_ERE. See [`data/README.md`](data/README.md) for access instructions.

## Mapping between the paper and the code

The class names match the terminology in Section 2.2 of the paper:

| Paper (Section 2.2) | Class | File | Description |
|---|---|---|---|
| Mention Expert | `MentionExpert` | `Models/Common_Experts/CommonExperts.py` | Encodes event mentions only |
| Sentence Expert | `SentenceExpert` | `Models/Common_Experts/SentenceExpert.py` | Encodes the sentences containing the two mentions |
| Context Expert | `ContextExpert` | `Models/Domain_Experts/ContextExpert.py` | Uses the title and a context window |
| Path Expert | `PathExpert` | `Models/Domain_Experts/PathExpert.py` | Uses paths of known relations between events |
| Graph Expert | `GraphExpert` | `Models/Domain_Experts/GraphExpert.py` | Models global structure with a GAT |
| Adaptive Expert Routing (AER) | `AER` | `Models/Router.py` | Dynamically selects and aggregates expert predictions |
| MER framework | `MERModel` | `Models/model.py` | Combines the five experts with AER |

## Repository structure

```text
MER/
├── README.md                # Project documentation (English / 中文)
├── mer/                     # Main model (RoBERTa; main results in the paper)
│   ├── main.py              # Two-phase training: experts at stages 1-5, router at stage 6
│   ├── parameter.py         # Hyperparameters
│   ├── data/                # Data loading and prompt construction
│   └── Models/              # MERModel: five experts and AER
│       ├── Common_Experts/  #   MentionExpert and SentenceExpert
│       ├── Domain_Experts/  #   ContextExpert, PathExpert, and GraphExpert
│       └── my_modules/      #   CGE (GAT, FocalLoss, etc.)
├── baselines/
│   ├── encoder_baselines/   # Encoder comparison baselines (PLM ablation)
│   │   ├── README.md        #   Encoder differences and usage
│   │   ├── deberta/         #   DeBERTa implementation with dynamic special-token IDs
│   │   ├── bert/            #   BERT implementation using AutoModel
│   │   └── ernie/           #   ERNIE implementation using AutoModel
│   ├── llm_prompt/          # GPT-3.5, GPT-4o-mini, and DeepSeek-V3.2 baselines
│   │                        #   Zero-shot and few-shot (1/3/5) prompting evaluation
│   └── llama_lora/          # Llama-2-7B instruction tuning with LoRA/QLoRA
├── data/
│   ├── samples/             # First event graph from each train/valid/test split
│   └── README.md            # Dataset access, construction, and citation
└── plot_analysis/           # Ablation and routing plots (expert distributions, top-k, etc.)
```

## Dependencies

```bash
pip install -r requirements.txt
```

Python 3.8+ and PyTorch. The model can run on a single GPU with 24 GB of memory using `batch_size=1`.

## Quick start

```bash
# 1. Obtain EGC-MAVEN; see data/README.md for the source and file placement.

# 2. Train MER (RoBERTa; stages 1-5 train the experts, and stage 6 trains the router).
cd mer
bash run.sh          # python main.py --t_lr 5e-6 --stage 6 --num_epoch 15

# 3. Run an encoder comparison baseline.
#    RoBERTa is the main encoder; see the individual directories for BERT/ERNIE.
cd ../baselines/encoder_baselines/deberta
bash run.sh
```

See `mer/parameter.py` for the main hyperparameters, including the number of experts (`num_experts`), routing settings (`top_k`, `top_p`, and `temperature`), the number of GAT heads (`num_heads`), and loss weighting (`weight_ratio`).

## Encoder comparison baselines

MER uses **RoBERTa-base** as its main encoder (`mer/`, corresponding to the main results table). **DeBERTa, BERT, and ERNIE** are encoder comparison baselines in the PLM ablation table. Encoder-specific code differences are concentrated in four areas; see [baselines/encoder_baselines/README.md](baselines/encoder_baselines/README.md):

1. Hugging Face model class (`RobertaForMaskedLM`, `DebertaForMaskedLM`, or `AutoModelForMaskedLM`).
2. Encoder body (`.roberta(...)`, `.deberta(...)`, `.bert(...)`, or `.ernie(...)`).
3. MLM output head (`.lm_head` for RoBERTa; `.cls` for DeBERTa/BERT/ERNIE).
4. Answer-space token IDs: the RoBERTa implementation uses fixed IDs `50265-50275`; the other implementations obtain IDs dynamically from the tokenizer for encoder adaptation.

- `mer/`: main RoBERTa implementation, corresponding to the main results table.
- `baselines/encoder_baselines/deberta/`: DeBERTa with four experts; stage 4 is skipped. See the [encoder baseline instructions](baselines/encoder_baselines/README.md).
- `baselines/encoder_baselines/bert/` and `baselines/encoder_baselines/ernie/`: BERT and ERNIE with five experts and a router, using stages 1-6.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{zhang2026mer,
  title     = {Multi-Expert Routing for Event Graph Completion},
  author    = {Zhang, Qing and Cai, Wei and Liang, Chao and Wang, Bang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

</details>

<details>
<summary><strong>中文</strong></summary>

官方代码库，对应论文 *Multi-Expert Routing for Event Graph Completion*（EMNLP 2026）。

MER（Multi-Expert Routing）是一个面向**异构事件图补全**（Event Graph Completion, EGC）的多专家框架：将异构事件图转换为结构化三元组序列，用提示学习范式在预训练语言模型上编码语义与结构信息，由 **AER**（Adaptive Expert Routing，自适应专家路由）模块针对每个查询事件对动态选择并加权聚合 5 个专家的预测，同时完成四类事件关系补全：

- **Sub-event**（子事件）
- **Causality**（因果）
- **Temporal**（时序）
- **Coreference**（共指）

5 个专家（论文 §2.2 命名）：**Mention Expert、Sentence Expert、Context Expert、Path Expert、Graph Expert**。

数据集为 **EGC-MAVEN**，由本团队前作 PLAF 论文（*Neural Networks*, 2026,
[10.1016/j.neunet.2026.108730](https://doi.org/10.1016/j.neunet.2026.108730)）构建，
基于 MAVEN_ERE；获取方式见 [`data/README.md`](data/README.md)。

## 模块命名对照（代码 ↔ 论文）

代码类名与论文 §2.2 完全一致：

| 论文（§2.2） | 代码类 | 代码文件 | 说明 |
|---|---|---|---|
| Mention Expert | `MentionExpert` | `Models/Common_Experts/CommonExperts.py` | 仅编码事件 mention |
| Sentence Expert | `SentenceExpert` | `Models/Common_Experts/SentenceExpert.py` | 编码包含两个 mention 的句子 |
| Context Expert | `ContextExpert` | `Models/Domain_Experts/ContextExpert.py` | Title + 上下文窗口 |
| Path Expert | `PathExpert` | `Models/Domain_Experts/PathExpert.py` | 事件间的已知关系路径 |
| Graph Expert | `GraphExpert` | `Models/Domain_Experts/GraphExpert.py` | GAT 建模全局结构 |
| Adaptive Expert Routing (AER) | `AER` | `Models/Router.py` | 动态选择 + 加权聚合 |
| MER 框架整体 | `MERModel` | `Models/model.py` | 五专家 + AER |

## 目录结构

```
MER/
├── README.md                # 项目说明（英文 / 中文）
├── mer/                     # 主模型代码（RoBERTa 主基线，论文主表结果）
│   ├── main.py              # 两阶段训练入口（stage 1-5 训练各专家，stage 6 训练路由器）
│   ├── parameter.py         # 全部超参数
│   ├── data/                # 数据加载与提示模板构造
│   └── Models/              # MERModel：5 专家 + AER 路由
│       ├── Common_Experts/  #   MentionExpert、SentenceExpert
│       ├── Domain_Experts/  #   ContextExpert、PathExpert、GraphExpert
│       └── my_modules/      #   CGE（GAT、FocalLoss 等）
├── baselines/
│   ├── encoder_baselines/   # 编码器对比基线（PLM Ablation）
│   │   ├── README.md        #   四编码器差异与运行说明
│   │   ├── deberta/         #   DeBERTa 版完整代码（动态获取特殊 token ID）
│   │   ├── bert/            #   BERT 版完整代码（AutoModel 统一方案）
│   │   └── ernie/           #   ERNIE 版完整代码（AutoModel 统一方案）
│   ├── llm_prompt/          # LLM 基线：GPT-3.5 / GPT-4o-mini / DeepSeek-V3.2
│   │                        #   zero-shot 与 few-shot(1/3/5) 线性化提示评测
│   └── llama_lora/          # LLM 基线：Llama2-7B LoRA/QLoRA 指令微调
├── data/
│   ├── samples/             # train/valid/test 首个事件图的格式样本
│   └── README.md            # 数据获取与构建说明（含数据集引用）
└── plot_analysis/           # 消融与路由分析画图脚本（专家路由分布、top-k、few-shot 曲线等）
```

## 环境依赖

```bash
pip install -r requirements.txt
```

Python 3.8+，PyTorch（单卡 24GB 即可运行，`batch_size=1`）。

## 快速开始

```bash
# 1. 获取数据（EGC-MAVEN，来源与放置路径见 data/README.md）

# 2. 训练 MER（RoBERTa 主实验；两阶段：stage 1-5 各专家，stage 6 路由器）
cd mer
bash run.sh          # python main.py --t_lr 5e-6 --stage 6 --num_epoch 15

# 3. 换用编码器对比基线
#    RoBERTa 为主基线；DeBERTa 完整代码可直接运行，BERT/ERNIE 见对应目录说明：
cd ../baselines/encoder_baselines/deberta
bash run.sh
```

主要超参数见 `mer/parameter.py`：专家数量 `num_experts`、路由 `top_k` / `top_p` / `temperature`、GAT 头数 `num_heads`、损失加权 `weight_ratio` 等。

## 编码器对比基线

MER 以 **RoBERTa-base** 为主基线（`mer/`，论文主表）；**DeBERTa / BERT / ERNIE** 作为
编码器对比基线（PLM Ablation 表）。代码差异集中在模型文件里的 4 处（详见 [baselines/encoder_baselines/README.md](baselines/encoder_baselines/README.md)）：

1. HuggingFace 模型类（`RobertaForMaskedLM` / `DebertaForMaskedLM` / `AutoModelForMaskedLM`）
2. 编码器主体属性（`.roberta(...)` / `.deberta(...)` / `.bert(...)` / `.ernie(...)`)
3. MLM 输出头属性（RoBERTa 是 `.lm_head`，DeBERTa/BERT/ERNIE 是 `.cls`）
4. 答案空间 token ID（RoBERTa 版为硬编码的 `50265-50275`；其余版本通过 tokenizer 动态获取，可迁移到任意编码器）

- `mer/` = RoBERTa 主基线（与论文主表对应）
- `baselines/encoder_baselines/deberta/` = DeBERTa（4 专家配置，跳过 stage 4，见其 README）
- `baselines/encoder_baselines/bert|ernie/` = BERT / ERNIE（5 专家 + Router，stage 1→6）

## 引用

如果本仓库对你的研究有帮助，请引用：

```bibtex
@inproceedings{zhang2026mer,
  title     = {Multi-Expert Routing for Event Graph Completion},
  author    = {Zhang, Qing and Cai, Wei and Liang, Chao and Wang, Bang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

</details>
