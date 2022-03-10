#!/bin/bash
#for id in 0 1 2 3 4 5 6 7 8 9 10
for id in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    CUDA_VISIBLE_DEVICES=1 python  run_mbpa.py \
    --nepoch 50 \
    --ntasks 1 \
    --bert_model ../ptm/bert-base-uncased \
    --bert_hidden_size 768 \
    --mbpa 5 \
    --trainmode $id \
    --startpoint $id \
    --inter 1
done

# 通过startpoint调整数据集，让当前领域位于队列第一个，然后设定任务为1，读取当前任务数据集，减少开销
# inter 是间隔 设置为1则为测试的时候全部都进行tuning
