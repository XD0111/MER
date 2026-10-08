# Encoder baselines

The main RoBERTa implementation is in [mer/](../../mer/). This directory contains the DeBERTa, BERT, and ERNIE variants used for encoder comparisons.

| Implementation | Default model | Expert stages | Checkpoint directory |
|---|---|---|---|
| [RoBERTa](../../mer/) | `roberta-base` | 1 Mention, 2 Sentence, 3 Context, 4 Path, 5 Graph; 6 AER | `checkpoint/` |
| [DeBERTa](deberta/) | `microsoft/deberta-base` | 1 Mention, 2 Sentence, 3 Context, 5 Graph; 6 AER | `checkpoint_deberta/` |
| [BERT](bert/) | `google-bert/bert-base-uncased` | 1 Mention, 2 Sentence, 3 Context, 4 Document, 5 Graph; 6 AER | `checkpoint_bert_1.0/` (default train ratio) |
| [ERNIE](ernie/) | `nghuyong/ernie-2.0-en` | 1 Mention, 2 Sentence, 3 Context, 4 Document, 5 Graph; 6 AER | `checkpoint_ernie/` |

These are separate experimental implementations. BERT/ERNIE use a Document Expert at stage 4; DeBERTa's training model omits the Path Expert. Do not substitute their stage mappings for the main MER mapping.

## Setup and data

Install the root [requirements.txt](../../requirements.txt). Run each variant from its own directory so that its local `Models` and `data` imports resolve correctly. Model names can be replaced with local pretrained-model directories using `--model_name`.

All variants use the processed dataset described in [data/README.md](../../data/README.md). Pass the three split paths explicitly to avoid copying the dataset:

```bash
# From baselines/encoder_baselines/deberta/
mkdir -p checkpoint_deberta out
python main.py --stage 1 --t_lr 5e-3 --num_epoch 15 \
  --train_data_path ../../../mer/data/train.json \
  --valid_data_path ../../../mer/data/valid.json \
  --test_data_path ../../../mer/data/test.json
```

Repeat for the expert stages in the table, then run stage 6. Each `run.sh` launches **only stage 6** and expects expert checkpoints to exist. Keep each encoder's checkpoints separate. BERT-specific instructions are in [bert/README.md](bert/README.md); ERNIE-specific instructions are in [ernie/README.md](ernie/README.md).

## Encoder adaptation

The implementations differ in the masked-language-model class, the encoder body (`.roberta`, `.deberta`, `.bert`, or `.ernie`), and the MLM head. RoBERTa uses fixed IDs for the relation tokens; the other encoder variants obtain these IDs from the tokenizer.

## DeBERTa evaluation and routing analysis

`deberta/eval_main.py` contains an alternative evaluation entry point. `deberta/analyze_expert_routing_ratio.py` exports routing statistics, with checkpoint paths configured through CLI arguments (including `--MOE_path` for a full model checkpoint).

The routing-analysis model instantiates five experts, including Path, whereas the DeBERTa training model uses four. Routing analysis therefore requires compatible five-expert checkpoints; four-expert training checkpoints cannot be used interchangeably.
