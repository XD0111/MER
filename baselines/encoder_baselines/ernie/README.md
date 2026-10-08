# ERNIE 编码器对比基线

MER 的 **ERNIE 2.0-en**（nghuyong 移植版）编码器版本（论文 Table 3 中 `MER ERNIE` 行，平均 F1 62.8）。
完整代码已入库，来源：`nlp-3090:/home/wzl/CaiW/MER/EGC_ernie`（AutoModel 统一版）。
**已验证**：RTX 4090 上 stage 1→6 全流程跑通（transformers 4.40 + torch 2.3）。

## 与主基线（mer/，RoBERTa）的差异

- 模型类：`AutoModelForMaskedLM` + `AutoTokenizer`，按权重 config 自动选择。
  nghuyong/ernie-2.0-en 的 config 为 `ernie` 类型，AutoModel 映射到 `ErnieForMaskedLM`，
  代码中主体属性为 `.ernie`、输出头 `.cls`；
- 答案空间 token ID 通过 tokenizer 动态获取，无硬编码；
- 专家阵容与 bert 版一致：Common / Title / Contextual / Document / Graph 共 5 个 + Router。

## 运行

```bash
# 权重：nghuyong/ernie-2.0-en（词表 30522，加 11 个提示 token 后 30533）
python main.py --model_name /path/to/ernie-2.0-en --stage 1 --num_epoch 15 ...
# ... 依次 stage 2/3/4/5 ...
python main.py --model_name /path/to/ernie-2.0-en --stage 6 --num_epoch 15 ...
```

环境变量 `CUDA_VISIBLE_DEVICES` 可覆盖默认卡号。

## 入库时修复的问题（相对原始副本）

1. `Models/` 下 8 个文件的模块级 `os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id`
   残留（`args` 在模块层未定义，导入即崩）→ 改为 `setdefault`；
2. 主体属性 `.bert(` → `.ernie(`（当前 nghuyong 权重 config 为 ernie 类型，
   AutoModel 返回 `ErnieForMaskedLM`，不存在 `.bert` 属性）。
