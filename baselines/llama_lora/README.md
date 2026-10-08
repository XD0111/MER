# Llama-2 LoRA/QLoRA baseline

This baseline converts event-relation data into instruction examples, trains a Llama-2-7B adapter, and evaluates generated relation labels. Each event-pair example becomes four classification prompts: sub-event, temporal, causal, and coreference.

## Setup

Use a separate Python 3.10+ environment and install this directory's dependencies:

```bash
pip install -r requirements.txt
export LLAMA2_7B_PATH=/absolute/path/to/Llama-2-7b-hf
```

`LLAMA2_7B_PATH` must point to a local model directory containing the model configuration, tokenizer, and weights. Install PyTorch for the CUDA environment in use. Run the following commands from `baselines/llama_lora/` in Bash.

## Prepare event-relation data

Obtain the full dataset following [data/README.md](../../data/README.md), then convert the splits:

```bash
export TRAIN_PATH=../../mer/data/train.json
export VALID_PATH=../../mer/data/valid.json
export TEST_PATH=../../mer/data/test.json
bash scripts/run_prepare_event.sh
```

The script writes `train.jsonl`, `valid.jsonl`, and `test.jsonl` to `data/processed_event/`. Rows include `prompt`, `completion`, `relation_type`, and `gold_label`. Candidate labels are:

| Relation | Candidates |
|---|---|
| Sub-event | `Belong to`, `Include`, `None` |
| Temporal | `Before`, `After`, `None` |
| Causal | `Cause`, `Caused by`, `None` |
| Coreference | `Cof`, `None` |

`data/raw/event_demo_*.json` contains small synthetic format examples. These examples illustrate the converter's input and are not experimental splits.

## Train

Edit [llama2_7b_lora_event_local.yaml](configs/llama2_7b_lora_event_local.yaml) for the desired training settings. The default uses 4-bit QLoRA, rank 16, and an output directory of `outputs/llama2_7b_lora_event_sft/`.

```bash
accelerate config
bash scripts/run_train_event_local.sh
```

For memory constraints, reduce `max_seq_length` or the LoRA target modules and adjust gradient accumulation. Save the final configuration with the experiment results.

## Inference and evaluation

After training:

```bash
bash scripts/run_test_event_local.sh
```

This loads the base model and adapter, writes `outputs/event_test_predictions.jsonl`, and invokes `src/evaluate_generation.py`. The evaluator reports exact match, character-level ROUGE-L, and candidate accuracy when candidate metadata is present. These generation metrics are separate from the relation-wise F1 calculation used by the encoder implementation.

## Zero-shot and few-shot inference

```bash
python src/fewshot_event_infer.py \
  --base_model "$LLAMA2_7B_PATH" \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --k 0 \
  --output_file outputs/event_zeroshot_predictions.jsonl \
  --max_new_tokens 16 --temperature 0
```

Set `--k` to the required number of training demonstrations for few-shot inference. `scripts/run_fewshot_event_local.sh` defaults to **4** demonstrations. Pass `--adapter_path outputs/llama2_7b_lora_event_sft` to use a trained adapter.

## Optional utilities

- `src/merge_lora.py`: merge an adapter into the base model.
- `src/prepare_data.py` and `scripts/run_prepare.sh`: convert generic instruction data; `data/raw_demo.jsonl` illustrates this separate format.
- `configs/llama2_7b_lora.yaml`, `scripts/run_train.sh`, and `scripts/run_infer.sh`: generic instruction-training workflow. Use the event-specific workflow above for event-relation experiments.

Full data, processed data, model weights, adapters, and prediction outputs are excluded from Git. The configuration files, source code, and format examples are included.
