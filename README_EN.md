# MER: Multi-Expert Routing for Event Graph Completion

[中文](README.md) | **English**

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
├── README.md                # Chinese documentation
├── README_EN.md             # English documentation
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
