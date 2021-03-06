#!/bin/bash

# =======================
#     PREPARE DATASET
# =======================

: '
----------------------------------------------------------------
    Author: Jiho Choi (jihochoi@snu.ac.kr)
        - https://github.com/JihoChoi
----------------------------------------------------------------
'

python ./dynamic-gcn/preparation/preprocess_dataset.py Twitter15 5
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 sequential 5
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 temporal 5




python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 sequential 10
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 temporal 10

python ./dynamic-gcn/preparation/preprocess_dataset.py Twitter15 10

python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter15 sequential 10
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter15 temporal 10



echo "-----------------------------------"
echo "        DATASET PREPARATION        "
echo "-----------------------------------"

# Preprocessing
python ./dynamic-gcn/preparation/preprocess_dataset.py Twitter16 3


# Snapshot Generation
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 sequential 3
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 temporal 3

python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter15 sequential 3
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter15 temporal 3



python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter15 sequential 5
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter15 temporal 5



echo "----------------------------"
echo "        TRAIN & TEST        "
echo "----------------------------"

python ./scripts/main.py Twitter16 temporal snapshot_5 additive_attention
python ./scripts/main.py Twitter16 temporal snapshot_5 self_attention


# Usage
python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3


# Usage
python ./dynamic-gcn/main.py -m GCN -ls dot_product -dn Twitter16 -dt sequential -sn 3



python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3


python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3



python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 3 --cuda cuda:0

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 3 --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 3 --cuda cuda:2


python ./dynamic-gcn/main.py --model GCN --learning-sequence mean \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3 --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence mean \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 3 --cuda cuda:2




python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3 --cuda cuda:0

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 3 --cuda cuda:0




python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 5 --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 5 --cuda cuda:0


python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 5 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 5 --cuda cuda:2

# TODO: 0728

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 5 --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 5 --cuda cuda:1

# TODO: 0728
# TODO: 0728
# TODO: 0728


python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 5 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 5 --cuda cuda:2




# TODO:

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 3 --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 3 --cuda cuda:2



# TODO:
# TODO:

# Attention

# TODO:

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 10 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 10 --cuda cuda:3


python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 10 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 10 --cuda cuda:3




# TODO: 0814 TODO: TODO: TODO: TODO: TODO: TODO: TODO:

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 10 --cuda cuda:0

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 10 --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 10 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 10 --cuda cuda:3













python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 5 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 5 --cuda cuda:3

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 5 --cuda cuda:1


# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE
python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 5 --cuda cuda:1

# TODO: 0813

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 10 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 10 --cuda cuda:3

# TODO: HERE
python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 10 --cuda cuda:2

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 10 --cuda cuda:3




# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE# HERE



python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3 --cuda cuda:0



# -------------------------------------
# WEIBO
# -------------------------------------

python ./dynamic-gcn/preparation/preprocess_dataset.py Weibo 3
python ./dynamic-gcn/preparation/prepare_snapshots_weibo.py Weibo sequential 3

python ./dynamic-gcn/main_weibo.py --model GCN --learning-sequence dot_product \
    --dataset-name Weibo --dataset-type sequential --snapshot-num 3

# 