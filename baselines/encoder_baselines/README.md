# 编码器对比基线（RoBERTa 主基线 / DeBERTa / BERT / ERNIE）

MER 主模型以 **RoBERTa-base** 为主基线（代码在仓库根目录 [`mer/`](../../mer/)，对应论文主表）。
本目录存放作为**对比基线**的其他预训练编码器版本：同一套 MER 框架、同一份训练/评测流程，
模型代码只有**少量、机械性**的差异。对应论文的 **PLM Ablation** 表
（`MER RoBERTa / MER DeBERTa / MER BERT / MER ERNIE` 四行）。

| 目录 | 编码器 | 状态 |
|---|---|---|
| [`../../mer/`](../../mer/) | RoBERTa（主基线） | ✅ 完整实验代码 |
| [`deberta/`](deberta/) | DeBERTa | ✅ 完整实验代码（动态 token 方案，含 eval_main.py） |
| [`bert/`](bert/) | BERT | 📄 适配说明（完整副本在合作者机器，派生步骤 2 处替换） |
| [`ernie/`](ernie/) | ERNIE | 📄 适配说明（同上） |

## 1. 差异在哪里

每个专家文件（`Models/Common_Experts/CommonExperts.py`、`TitleExpert.py`、
`Models/Domain_Experts/ContextualExpert.py`、`PathExpert.py`、`GraphExpert.py`、
`Models/Router.py`、`Models/model.py`）中，编码器相关的代码有 4 处差异：

| 位置 | RoBERTa 版 | DeBERTa 版 | BERT / ERNIE 版（同 DeBERTa 方案） |
|---|---|---|---|
| ① 模型类 | `from transformers import RobertaForMaskedLM` | `from transformers import DebertaForMaskedLM` | `BertForMaskedLM` / `ErnieForMaskedLM` |
| ② 编码器主体 | `self.RoBERTa_MLM_for_schema.roberta(...)` | `self.Deberta_MLM_for_schema.deberta(...)` | `.bert(...)` / `.ernie(...)` |
| ③ MLM 输出头 | `.lm_head(...)` | `.cls(...)` | `.cls(...)` |
| ④ 词嵌入路径 | `.roberta.embeddings.word_embeddings.weight` | `.deberta.embeddings.word_embeddings.weight` | `.bert.embeddings...` / `.ernie.embeddings...` |

另外一处**关键适配**是答案空间（answer space）token ID：

- **RoBERTa 版**：特殊提示 token（`<ca0>~<ca2>`、`<te0>~<te2>`、`<su0>~<su2>`、`<co0>~<co1>`）
  加入词表后的 ID 是**硬编码**的：

  ```python
  answer_space_sub  = [50273, 50274, 50275]
  answer_space_temp = [50268, 50269, 50270]
  answer_space_cau  = [50265, 50266, 50267]
  answer_space_cof  = [50271, 50272]
  ```

- **DeBERTa 版**（本目录 [`deberta/`](deberta/)）：改为**通过 tokenizer 动态获取**，
  因此天然适配任意词表的编码器：

  ```python
  self.tokenizer = tokenizer   # __init__ 增加 tokenizer 参数
  self.answer_space_sub  = [tokenizer.convert_tokens_to_ids(t) for t in ('<su0>', '<su1>', '<su2>')]
  self.answer_space_temp = [tokenizer.convert_tokens_to_ids(t) for t in ('<te0>', '<te1>', '<te2>')]
  self.answer_space_cau  = [tokenizer.convert_tokens_to_ids(t) for t in ('<ca0>', '<ca1>', '<ca2>')]
  self.answer_space_cof  = [tokenizer.convert_tokens_to_ids(t) for t in ('<co0>', '<co1>')]
  ```

其余差异均为表面性的：`os.environ['CUDA_VISIBLE_DEVICES']` 取值、`parameter.py`
中 `--model_name` 指向的本地权重路径、`--vocab_size`（RoBERTa/DeBERTa 为 50265+新增 token）。

## 2. 代码副本来源

- **RoBERTa / DeBERTa**：正式实验代码，分别来自本地工作副本
  `20250601-base-V1.33/new/` 与 `20250601-base-V1.33/EGC/`（详见 [docs/CODEMAP.md](../../docs/CODEMAP.md)）。
- **BERT / ERNIE**：权重曾存放于 `nlp-3090:/home/wzl/prompt-learning/PLMs/{BERT, ernie-2.0-en}`
  与 `5000:/home/luoq/Lc_TEMP/{DeBERTa, ernie-2.0-en}`；**完整代码副本未在可访问机器上检索到**
  （应在合作者账号 gp3_liangc 的机器上），两个目录目前提供的是与 DeBERTa 版同构的
  派生方法说明。

## 3. 从 DeBERTa 版派生 BERT / ERNIE 版

以 BERT 为例（ERNIE 同理，见 [`bert/README.md`](bert/README.md)、[`ernie/README.md`](ernie/README.md)）：

1. 复制 `deberta/` 整个目录为 `bert/` 的代码。
2. 在 `Models/` 与 `data/` 下全部 `.py` 文件做全局替换：
   `DebertaForMaskedLM → BertForMaskedLM`、`Deberta_MLM → Bert_MLM`、
   `.deberta( → .bert(`、`.deberta.embeddings → .bert.embeddings`
   （`.cls(` 输出头不需要改）。
3. `parameter.py` 中把 `--model_name` 指向对应编码器权重目录；
   `--vocab_size` 按新增提示 token 后的实际词表大小设置。
4. `data/` 下预处理脚本使用同一 tokenizer 动态获取提示 token ID，无需其他修改。

> 注意：上述派生步骤得到的是**未经运行验证**的代码；论文中 BERT / ERNIE 两行结果
> 来自合作者机器上的原始副本，若需严格复现请以原始副本为准。
