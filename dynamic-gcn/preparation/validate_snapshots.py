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

    # --------------------------
    #         INIT PATHS
    # --------------------------
    paths = {}
    if dataset in ['Twitter15', 'Twitter16']:
        # paths['raw'] = './data/raw/rumor_detection_acl2017/'
        # paths['raw_label'] = os.path.join(paths['raw'], dataset.lower(), 'label.txt')
        # paths['raw_tree'] = os.path.join(paths['raw'], dataset.lower(), 'tree/')
        paths['resource_label'] = './resources/{0}/{0}_label_all.txt'.format(dataset)
        paths['resource_tree'] = './resources/{0}/data.TD_RvNN.vol_5000.txt'.format(dataset)
        # paths['timestamps'] = './data/timestamps/{}/timestamps.txt'.format(dataset)
        paths['timestamps_trim'] = './data/timestamps/{}/timestamps_trim.txt'.format(dataset)
        paths['sequential_snapshots'] = './data/timestamps/{}/sequential_snapshots_{:02}.txt'.format(dataset, snapshot_num)
        paths['temporal_snapshots'] = './data/timestamps/{}/temporal_snapshots_{:02}.txt'.format(dataset, snapshot_num)
    elif dataset in ['Weibo']:
        exit()
    else:
        exit()
    print_dict(paths)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
