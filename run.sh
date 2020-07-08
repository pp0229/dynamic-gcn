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


