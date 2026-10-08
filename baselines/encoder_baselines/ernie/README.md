# ERNIE encoder baseline

This implementation uses `AutoModelForMaskedLM` and defaults to `nghuyong/ernie-2.0-en`. Relation-token IDs are obtained from the tokenizer.

The stages are **1 Mention, 2 Sentence, 3 Context, 4 Document, 5 Graph, 6 AER**. Unlike the main RoBERTa implementation, this variant uses Document at stage 4.

## Training

Run from this directory after installing the root requirements and obtaining the dataset:

```bash
mkdir -p checkpoint_ernie out
set -e
for stage in 1 2 3 4 5; do
  python main.py --model_name nghuyong/ernie-2.0-en \
    --train_data_path ../../../mer/data/train.json \
    --valid_data_path ../../../mer/data/valid.json \
    --test_data_path ../../../mer/data/test.json \
    --stage "$stage" --t_lr 5e-3 --num_epoch 15
done

# Run after all five experts have finished successfully.
python main.py \
  --train_data_path ../../../mer/data/train.json \
  --valid_data_path ../../../mer/data/valid.json \
  --test_data_path ../../../mer/data/test.json \
  --stage 6 --t_lr 5e-3 --num_epoch 15
```

Checkpoints are saved to `checkpoint_ernie/`. `bash run.sh` launches only stage 6 and requires the five expert checkpoints. `--model_name` also accepts a local model directory. See [parameter.py](parameter.py) for the full configuration.
