# BERT 编码器对比基线

MER 的 **BERT-base-uncased** 编码器版本（论文 Table 3 中 `MER BERT` 行，平均 F1 58.6）。
完整代码已入库，来源：`nlp-3090:/home/wzl/CaiW/MER/EGC_bert_new`（AutoModel 统一版）。
**已验证**：RTX 4090 上 stage 1→6 全流程跑通（transformers 4.40 + torch 2.3）。

## 与主基线（mer/，RoBERTa）的差异

- 模型类：`AutoModelForMaskedLM` + `AutoTokenizer(use_fast=False)`，按权重 config 自动选择；
- 答案空间 token ID 通过 tokenizer 动态获取（`data/processe_data.py` 预处理时
  `add_tokens` 并更新 `args.vocab_size`），无硬编码；
- 专家阵容：Common / Title / Contextual / **Document** / Graph 共 5 个 + Router（stage 1-6）。

## 运行

```bash
# 权重：google-bert/bert-base-uncased（词表 30522，加 11 个提示 token 后 30533）
python main.py --model_name /path/to/bert-base-uncased --stage 1 --num_epoch 15 ...
# ... 依次 stage 2/3/4/5 ...
python main.py --model_name /path/to/bert-base-uncased --stage 6 --num_epoch 15 ...
```

注意：main.py 末尾有一处整模测试（需要全部 5 个专家的 checkpoint）。训练 stage 1-5 时
可设 `MER_SKIP_FINAL_TEST=1` 跳过它，stage 6 时正常执行（此时 checkpoint 已齐）。

环境变量 `CUDA_VISIBLE_DEVICES` 可覆盖默认卡号（默认沿用原实验机的 1 号卡）。
