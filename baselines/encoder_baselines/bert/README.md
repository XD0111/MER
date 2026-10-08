# BERT encoder baseline

This implementation uses `AutoModelForMaskedLM` and defaults to `google-bert/bert-base-uncased`. Relation-token IDs are obtained from the tokenizer.

The stages are **1 Mention, 2 Sentence, 3 Context, 4 Document, 5 Graph, 6 AER**. Unlike the main RoBERTa implementation, this variant uses Document at stage 4.

## Training

Run from this directory after installing the root requirements and obtaining the dataset:

```bash
mkdir -p checkpoint_bert_1.0 out
set -e
for stage in 1 2 3 4 5; do
  MER_SKIP_FINAL_TEST=1 python main.py \
    --model_name google-bert/bert-base-uncased \
    --train_data_path ../../../mer/data/train.json \
    --valid_data_path ../../../mer/data/valid.json \
    --test_data_path ../../../mer/data/test.json \
    --stage "$stage" --t_lr 5e-6 --num_epoch 15
done

# Run after all five experts have finished successfully.
python main.py \
  --train_data_path ../../../mer/data/train.json \
  --valid_data_path ../../../mer/data/valid.json \
  --test_data_path ../../../mer/data/test.json \
  --stage 6 --t_lr 5e-6 --num_epoch 15
```

`MER_SKIP_FINAL_TEST=1` skips the additional full-model test at the end of expert training, which otherwise requires all expert checkpoints. `bash run.sh` launches only stage 6 with its own preset parameters.

The default checkpoint directory is `checkpoint_bert_1.0/`. Other train ratios and ablations change the save directory; update the checkpoint-loading arguments accordingly. `--model_name` also accepts a local model directory. See [parameter.py](parameter.py) for GPU, ablation, and training settings.
