import os
import sys
import time
import numpy as np
import json

sys.path.insert(0, './dynamic-gcn/')
from utils import print_dict

def main():
    # -------------------------------
    #         PARSE ARGUMENTS
    # -------------------------------
    arg_names = ['command', 'dataset_name', 'dataset_type', 'snapshot_num']
    if len(sys.argv) != 4:
        print("Please check the arguments.\n")
        print("Example usage:")
        print("python ./.../prepare_snapshots.py Twitter16 sequential 3")
        exit()
    args = dict(zip(arg_names, sys.argv))
    dataset = args['dataset_name']
    dataset_type = args['dataset_type']
    snapshot_num = int(args['snapshot_num'])
    print_dict(args)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
