#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python  run_mbpa.py \
    --nepoch 50 \
    --ntasks 21 \
    --bert_model ../ptm/bert-base-uncased \
    --bert_hidden_size 768 \
    --mbpa 5  \
    --train_batch_size 64

 
