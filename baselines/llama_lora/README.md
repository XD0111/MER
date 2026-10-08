# Llama-2-7B LoRA/QLoRA 事件关系指令微调工程

本工程已经适配你现有的事件图 / 事件关系数据处理逻辑，目标是把原始数据转换成 Llama2 可训练的 `prompt/completion` 指令微调格式，然后用 LoRA 或 QLoRA 训练。

核心流程：

1. 读取原始 train / valid / test JSON 或 JSONL。
2. 根据事件图构造 `mention_schema`。
3. 将每个事件对扩展成 4 个指令分类样本：Sub-event、Temporal、Causal、Coreference。
4. 输出 TRL `SFTTrainer` 支持的 `prompt/completion` JSONL。
5. 加载本地 Llama-2-7B 权重，训练 LoRA adapter。
6. 测试时加载本地 base model + LoRA adapter，生成并解析候选标签。

## 1. 目录结构

```text
llama2_lora_sft_project/
├── configs/
│   ├── llama2_7b_lora.yaml
│   └── llama2_7b_lora_event_local.yaml
├── data/
│   ├── raw_demo.jsonl
│   └── raw/
│       ├── event_demo_train.json
│       ├── event_demo_valid.json
│       └── event_demo_test.json
├── scripts/
│   ├── run_prepare_event.sh
│   ├── run_train_event_local.sh
│   ├── run_test_event_local.sh
│   └── run_fewshot_event_local.sh
└── src/
    ├── event_relation_prompt.py
    ├── prepare_event_relation_data.py
    ├── train_lora_sft.py
    ├── infer.py
    ├── evaluate_generation.py
    ├── fewshot_event_infer.py
    └── merge_lora.py
```

## 2. 环境安装

建议先按你的服务器 CUDA 版本安装 PyTorch，然后安装其余依赖。

```bash
conda create -n llama2-lora python=3.10 -y
conda activate llama2-lora
pip install -r requirements.txt
```

如果你已经有 CUDA 12.2 对应的 PyTorch 环境，也可以直接在原环境里安装 `requirements.txt`。

## 3. 设置本地 Llama2 权重路径

你已经把 Llama2 权重下载到本地，所以不需要再从 Hugging Face 下载。运行前设置：

```bash
export LLAMA2_7B_PATH=/absolute/path/to/Llama-2-7b-hf
```

这个目录里通常应包含：

```text
config.json
generation_config.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json
```

如果你的文件名略有不同也没关系，只要 `AutoTokenizer.from_pretrained()` 和 `AutoModelForCausalLM.from_pretrained()` 能从该目录加载即可。

## 4. 原始数据格式

适配的是你原项目中类似下面的结构：

```json
[
  [
    {
      "node": {
        "E1": {"mention": "explosion", "sent_id": 0, "sentence": "The explosion damaged the building .", "location": [1, 2], "type": "event"},
        "E2": {"mention": "damage", "sent_id": 0, "sentence": "The explosion damaged the building .", "location": [2, 3], "type": "event"}
      },
      "relation": {
        "SUB_EVENT": [],
        "TEMPORAL": [],
        "CAUSAL": [["E1", "E2"]],
        "COF_EVENT": []
      },
      "sentences": {"0": "The explosion damaged the building ."}
    },
    ["E1", "E2"],
    [2, 2, 0, 1]
  ]
]
```

其中最后的 `[2, 2, 0, 1]` 分别对应：

```text
Sub-event label index
Temporal label index
Causal label index
Coreference label index
```

候选标签映射为：

```text
Sub-event:    Belong to / Include / None
Temporal:     Before / After / None
Causal:       Cause / Caused by / None
Coreference:  Cof / None
```

## 5. 数据处理

先把你的原始数据路径传进去：

```bash
export TRAIN_PATH=/path/to/train.json
export VALID_PATH=/path/to/valid.json
export TEST_PATH=/path/to/test.json
bash scripts/run_prepare_event.sh
```

或直接运行：

```bash
python src/prepare_event_relation_data.py \
  --train_path /path/to/train.json \
  --valid_path /path/to/valid.json \
  --test_path /path/to/test.json \
  --output_dir data/processed_event \
  --single 1 \
  --seed 42
```

输出：

```text
data/processed_event/train.jsonl
data/processed_event/valid.jsonl
data/processed_event/test.jsonl
```

每一行是：

```json
{"prompt": "Description: ...\nCandidates: ...\nQuestion: ...\nAnswer: ", "completion": "Cause</s>", "relation_type": "Causal", "gold_label": "Cause"}
```

## 6. 训练 LoRA / QLoRA

配置文件：

```text
configs/llama2_7b_lora_event_local.yaml
```

默认使用 QLoRA：

```yaml
model_name_or_path: ${LLAMA2_7B_PATH}
train_file: data/processed_event/train.jsonl
eval_file: data/processed_event/valid.jsonl
output_dir: outputs/llama2_7b_lora_event_sft
use_qlora: true
load_in_16bit: true
lora_r: 16
lora_alpha: 32
```

开始训练：

```bash
export LLAMA2_7B_PATH=/absolute/path/to/Llama-2-7b-hf
accelerate config
bash scripts/run_train_event_local.sh
```

或：

```bash
accelerate launch src/train_lora_sft.py --config configs/llama2_7b_lora_event_local.yaml
```

训练完成后，LoRA adapter 默认保存到：

```text
outputs/llama2_7b_lora_event_sft
```

## 7. 测试 LoRA 模型

```bash
bash scripts/run_test_event_local.sh
```

或直接：

```bash
python src/infer.py \
  --base_model "$LLAMA2_7B_PATH" \
  --adapter_path outputs/llama2_7b_lora_event_sft \
  --input_file data/processed_event/test.jsonl \
  --output_file outputs/event_test_predictions.jsonl \
  --max_new_tokens 16 \
  --temperature 0

python src/evaluate_generation.py --pred_file outputs/event_test_predictions.jsonl
```

评估脚本会输出：

```json
{
  "n": 100,
  "exact_match": 0.42,
  "rouge_l_char": 0.68,
  "candidate_accuracy": 0.71
}
```

其中 `candidate_accuracy` 是更适合这个任务的指标，因为它会把模型生成文本解析到候选标签，例如 `Cause`, `Caused by`, `None`。

## 8. Few-shot / zero-shot 本地测试

如果你还想复现原来 `main_2.py` 那种 API 测试逻辑，但改成本地 Llama2，可以用：

```bash
export LLAMA2_7B_PATH=/absolute/path/to/Llama-2-7b-hf
export TRAIN_PATH=/path/to/train.json
export TEST_PATH=/path/to/test.json
bash scripts/run_fewshot_event_local.sh
```

默认 `k=4`。如果要 zero-shot：

```bash
python src/fewshot_event_infer.py \
  --base_model "$LLAMA2_7B_PATH" \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --k 0 \
  --output_file outputs/event_zeroshot_predictions.jsonl \
  --max_new_tokens 16 \
  --temperature 0
```

如果想测试 LoRA 后再 few-shot，可以额外加：

```bash
--adapter_path outputs/llama2_7b_lora_event_sft
```

## 9. 合并 LoRA 权重，可选

```bash
python src/merge_lora.py \
  --base_model "$LLAMA2_7B_PATH" \
  --adapter_path outputs/llama2_7b_lora_event_sft \
  --output_dir outputs/llama2_7b_lora_event_merged
```

## 10. 文件说明

### `src/event_relation_prompt.py`

这是本次适配的核心文件，整合了原项目里的：

- `get_mention_schema`
- `trans_llm`
- 候选标签定义
- prompt 构造
- few-shot prompt 拼接
- 生成结果到候选标签的解析

### `src/prepare_event_relation_data.py`

把原始事件图数据转换成 Llama2 指令微调数据。

### `src/fewshot_event_infer.py`

本地模型 few-shot / zero-shot 测试脚本，不再调用外部 API，也不包含 API key。

### `src/train_lora_sft.py`

保留原来的 TRL `SFTTrainer + PEFT LoRA` 训练方式，并增加了 `${LLAMA2_7B_PATH}` 环境变量解析和本地模型加载支持。

## 11. 常见问题

### 显存不够

优先改：

```yaml
max_seq_length: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
use_qlora: true
```

如果仍然 OOM，把 LoRA 目标模块改轻量：

```yaml
target_modules: [q_proj, v_proj]
```

### 输出不是候选标签

这个任务最好让模型短输出：

```bash
--max_new_tokens 16 --temperature 0
```

prompt 里已经要求：

```text
Please respond with a single word or phrase.
```

### 训练集数量为什么变成 4 倍

因为每个事件对都被拆成 4 个关系判断任务：Sub-event、Temporal、Causal、Coreference。这与原 `trans_llm` 的处理方式一致。
