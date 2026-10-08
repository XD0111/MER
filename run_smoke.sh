#!/bin/bash
# MER 类名重命名后的全量验证：四编码器 stage 1→6
P=/home/bbx/anaconda3/envs/chatglm/bin/python
D=/home/bbx/user_cw/MER
TD=$D/data/smoke/train.json
VD=$D/data/smoke/valid.json
SD=$D/data/smoke/test.json
SUM=$D/smoke_summary7.txt
: > $SUM

run_stages() {
  name=$1; dir=$2; model=$3; skip=$4; stages=$5
  cd $D/$dir || { echo "$name DIR_FAIL" >> $SUM; return; }
  mkdir -p checkpoint checkpoint_ernie checkpoint_debert checkpoint_deberta out
  for s in $stages; do
    extra=""
    [ "$skip" = "yes" ] && [ "$s" -lt 6 ] && extra="MER_SKIP_FINAL_TEST=1"
    env CUDA_VISIBLE_DEVICES=0 $extra timeout 1800 $P main.py \
      --model_name $model \
      --train_data_path $TD --valid_data_path $VD --test_data_path $SD \
      --num_epoch 1 --stage $s --print_frequency 1 \
      > $D/smoke_${name}_stage${s}.log 2>&1
    echo "===== $name stage$s exit=$? $(date +%T) =====" >> $SUM
  done
}

run_stages roberta mer                                              /home/bbx/user_cw/PLMs/RoBERTa/RoBERTaForMaskedLM/roberta-base no  "1 2 3 4 5 6"
run_stages deberta baselines/encoder_baselines/deberta              /home/bbx/user_cw/PLMs/DeBERTa/deberta-base                   no  "1 2 3 5 6"
run_stages bert     baselines/encoder_baselines/bert                /home/bbx/user_cw/PLMs/BERT/bert-base-uncased                 yes "1 2 3 4 5 6"
run_stages ernie    baselines/encoder_baselines/ernie               /home/bbx/user_cw/PLMs/ERNIE/ernie-2.0-en                     no  "1 2 3 4 5 6"
echo "ALL DONE $(date +%T)" >> $SUM
