#!/usr/bin/env bash

TASK_ID=0
echo $TASK_ID
export MASTER_PORT=$((12073 + TASK_ID))
echo $MASTER_PORT

cd ~/projects/fairseq

#method="topk"
#trimmedreg=0.0
#version=10

##CUDA_VISIBLE_DEVICES=0 python train.py \
#torchrun --nproc_per_node=8 --nnodes=1 --master_addr=localhost --master_port=$MASTER_PORT \
#$(which fairseq-train) \
#data-bin-joined-dict/iwslt14.tokenized.de-en --inference-level 0 --arch moe_transformer_iwslt_de_en \
#--share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#--dropout 0.1 --weight-decay 0.0001 --max-epoch 100 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#--max-tokens 4000 \
#--update-freq 1 \
#--ddp-backend legacy_ddp \
#--log-format json --log-interval 10 --num-workers 16 \
#--save-interval 1 --keep-last-epochs 10 \
#--seed 1 \
#--task translation_moe --user-dir moe --route-method ${method} --k 2 --trimmed-lasso-reg ${trimmedreg} --save-dir "moe_iwslt14_de_en_${method}_checkpoints_${trimmedreg}_v${version}" \
#--num-experts 16 \
#--encoder-layers 6 --decoder-layers 6 \
#--encoder-attention-heads 16 --decoder-attention-heads 16 \
#--encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
#--valid-subset "valid,test" \
#--eval-bleu \
#--eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
#--eval-bleu-detok moses \
#--eval-bleu-remove-bpe \
#--eval-bleu-print-samples \
#--best-checkpoint-metric bleu \
#--maximize-best-checkpoint-metric \
#|& tee -a "moe_iwslt14_de_en_${method}_checkpoints_${trimmedreg}_v${version}.txt"

#method="moesart"
#trimmedreg=0.001
#version=10

#torchrun --nproc_per_node=8 --nnodes=1 --master_addr=localhost --master_port=$MASTER_PORT \
#$(which fairseq-train) \
#data-bin-joined-dict/iwslt14.tokenized.de-en --inference-level 0 --arch moe_transformer_iwslt_de_en \
#--share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#--dropout 0.1 --weight-decay 0.0001 --max-epoch 100 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#--max-tokens 4000 \
#--update-freq 1 \
#--ddp-backend legacy_ddp \
#--log-format json --log-interval 10 --num-workers 16 \
#--save-interval 1 --keep-last-epochs 10 \
#--seed 1 \
#--task translation_moe --user-dir moe --route-method ${method} --k 2 --trimmed-lasso-reg ${trimmedreg} --save-dir "moe_iwslt14_de_en_${method}_checkpoints_${trimmedreg}_v${version}" \
#--num-experts 16 \
#--encoder-layers 6 --decoder-layers 6 \
#--encoder-attention-heads 16 --decoder-attention-heads 16 \
#--encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
#--valid-subset "valid,test" \
#--eval-bleu \
#--eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
#--eval-bleu-detok moses \
#--eval-bleu-remove-bpe \
#--eval-bleu-print-samples \
#--best-checkpoint-metric bleu \
#--maximize-best-checkpoint-metric \
#|& tee -a "moe_iwslt14_de_en_${method}_checkpoints_${trimmedreg}_v${version}.txt"

method="none"
trimmedreg=0.0
version=10


torchrun --nproc_per_node=8 --nnodes=1 --master_addr=localhost --master_port=$MASTER_PORT \
$(which fairseq-train) \
data-bin-joined-dict/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en \
--share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--dropout 0.1 --weight-decay 0.0001 --max-epoch 100 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4000 \
--update-freq 1 \
--ddp-backend legacy_ddp \
--log-format json --log-interval 10 --num-workers 16 \
--save-interval 1 --keep-last-epochs 10 \
--seed 1 \
--task translation --user-dir moe --save-dir "moe_iwslt14_de_en_${method}_checkpoints_${trimmedreg}_v${version}" \
--encoder-layers 6 --decoder-layers 6 \
--encoder-attention-heads 16 --decoder-attention-heads 16 \
--encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
--valid-subset "valid,test" \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
|& tee -a "moe_iwslt14_de_en_${method}_checkpoints_${trimmedreg}_v${version}.txt"
