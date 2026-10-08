# API prompting baselines

This directory contains API-based zero-shot and few-shot event-relation baselines. Install the root requirements and obtain the processed data described in [data/README.md](../../data/README.md).

| Script | Setting |
|---|---|
| `main_1.py` | Sequential zero-shot inference |
| `main_2.py` | Asynchronous zero-shot inference |
| `main_fewshot.py` | Asynchronous few-shot inference, default 1 demonstration |
| `main_fewshot_3.py` | Asynchronous few-shot inference, default 3 demonstrations |
| `main_fewshot_5.py` | Asynchronous few-shot inference, default 5 demonstrations |
| `txt_result.py` | Relation-wise evaluation of saved predictions |

Run from this directory. Credentials are read from `OPENAI_API_KEY`. The scripts specify the model and API endpoint in their request code; set those to the provider/model used for the experiment before running. `--model_name` in `parameter.py` is a legacy local-model setting and does not select the API model.

```bash
export OPENAI_API_KEY="your-api-key"
export FEW_SHOT_K=3
export CONCURRENCY_LIMIT=5
python main_fewshot_3.py \
  --train_data_path ../../mer/data/train.json \
  --test_data_path ../../mer/data/test.json
```

Few-shot scripts accept `FEW_SHOT_K` (0 for zero-shot) and `FEW_SHOT_STRATEGY`; their random sampling uses the configured seed. Training examples must come from the training split.

Predictions are written to `test_file*.txt`, and logs are written to `out/`. Set `input_file` at the bottom of `txt_result.py` to a prediction file before running it. These outputs and credentials are excluded from Git.
