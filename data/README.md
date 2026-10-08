# 数据说明

## EGC-MAVEN 数据集

EGC-MAVEN 由本团队前作 **PLAF** 论文（Neural Networks 2026）构建并发布，MER 直接沿用：

> Chao Liang, Bang Wang, Chuanhong Zhan, Wei Xiang.
> **Multiplex Graph Prompt Learning and Attentive Fusion for Event Graph Completion.**
> *Neural Networks*, vol. 199, Article 108730, 2026. DOI: [10.1016/j.neunet.2026.108730](https://doi.org/10.1016/j.neunet.2026.108730)

- PLAF 论文官方代码仓库：<https://github.com/ChaoLiang-HUST/PLAF>
- 数据集基于 MAVEN_ERE（EMNLP 2022）构建：抽取事件并构建异构事件图
  （Sub-event / Causality / Temporal / Coreference 四种关系边），再转换为模型输入格式。

规模（处理完成后，MER 与 PLAF 使用同一划分）：

| split | 事件图数量 |
|---|---|
| train | 10,991 |
| valid | 3,589 |
| test | 3,601 |

`samples/` 下给出每个 split 第一个事件图的格式样本。

## 获取方式

### 方式一：引用并下载 PLAF 发布的数据（推荐）

数据集随 PLAF 工作发布，请优先通过 PLAF 论文 / 官方仓库获取
（<https://github.com/ChaoLiang-HUST/PLAF>；若仓库暂未附数据文件，
请通过论文联系作者获取）。

### 方式二：直接下载处理好的数据

处理后的 `train/valid/test.json`（约 550MB，单文件超 GitHub 100MB 限制）不入库，
建议随本仓库的 **GitHub Release 附件**分发，下载后放置为：

```
mer/data/train.json
mer/data/valid.json
mer/data/test.json
```

> 数据集构建（MAVEN_ERE → 事件图 → EGC-MAVEN）代码未随本仓库分发，如有需要请联系作者。

## 训练前的缓存

`mer/main.py` 首次运行会将 JSON 解析为 token 化缓存（`data_cache.pth`，约 2GB），
之后各阶段训练直接读缓存。`--stage` 决定训练哪个专家（1-5）或路由器（6）。

## 格式

每个事件图是一个列表，元素依次为：文档标题、事件节点表（mention/type/sentence/location）、
邻接表（按关系类型分组的边）等。字段含义以 `mer/data/processe_data.py` 中的
schema 构造逻辑为准；`data/samples/train_sample.json` 可直接对照查看。
