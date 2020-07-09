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




echo "----------------------------"
echo "        TRAIN & TEST        "
echo "----------------------------"

python ./scripts/main.py Twitter16 temporal snapshot_5 additive_attention
python ./scripts/main.py Twitter16 temporal snapshot_5 self_attention


# Usage
python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3



python ./dynamic-gcn/main.py -m GCN -ls dot_product -dn Twitter16 -dt sequential -sn 3



python ./dynamic-gcn/main.py --model GCN --attention dot_product \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3
