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

python ./dynamic-gcn/preparation/preprocess_dataset.py Twitter16 3

