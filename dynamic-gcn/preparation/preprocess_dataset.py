import sys
import os
import re
import time
import json
import random

sys.path.insert(0, './dynamic-gcn')
from utils import ensure_directory
from utils import load_json_file
from utils import save_json_file
from utils import print_dict


arg_names = ['command', 'dataset_name', 'snapshot_num']
args = dict(zip(arg_names, sys.argv))
dataset_name = args['dataset_name']
snapshot_num = int(args['snapshot_num'])
print_dict(args)


# ----------------------
#     READ ARGUMENTS
# ----------------------



if dataset_name in ['Twitter15', 'Twitter16']:
    dataset_name_lower = dataset_name.lower()
    RAW_PATH = './data/raw/rumor_detection_acl2017'
    RAW_LABEL_PATH = '{}/{}/label.txt'.format(RAW_PATH, dataset_name_lower)
    RAW_TREE_DIR_PATH = '{}/{}/tree/'.format(RAW_PATH, dataset_name_lower)
    RESOURCE_LABEL_PATH = './resources/{0}_label_All.txt'.format(dataset_name)
    RESOURCE_TREE_PATH = './resources/data.TD_RvNN.vol_5000.txt'.format(dataset_name)
elif dataset_name in ['Weibo']:
    exit()
else:
    exit()

print(RAW_PATH)
print(RAW_LABEL_PATH)
print(RAW_TREE_DIR_PATH)
print(RESOURCE_LABEL_PATH)
print(RESOURCE_TREE_PATH)



print()
exit()



if args['dataset_name'] == 'Twitter15':
    RAW_LABEL_PATH = './data/raw/rumor_detection_acl2017/twitter15/label.txt'
    RAW_TREE_PATH = './data/raw/rumor_detection_acl2017/twitter15/tree/'
    RESOURCE_LABEL_PATH = './resources/BiGCN/Twitter15/Twitter15_label_All.txt'
    RESOURCE_TREE_PATH = './resources/BiGCN/Twitter15/data.TD_RvNN.vol_5000.txt'
elif args['dataset_name'] == 'Twitter16':
    RAW_LABEL_PATH = './data/raw/rumor_detection_acl2017/twitter16/label.txt'
    RAW_TREE_PATH = './data/raw/rumor_detection_acl2017/twitter16/tree/'
    RESOURCE_LABEL_PATH = './resources/BiGCN/Twitter16/Twitter16_label_All.txt'
    RESOURCE_TREE_PATH = './resources/BiGCN/Twitter16/data.TD_RvNN.vol_5000.txt'
else:
    exit()


"""
# Twitter 15
RAW_LABEL_PATH = './data/raw/rumor_detection_acl2017/twitter15/label.txt'
RAW_TREE_PATH = './data/raw/rumor_detection_acl2017/twitter15/tree/'
RESOURCE_LABEL_PATH = './resources/BiGCN/Twitter15/Twitter15_label_All.txt'
RESOURCE_TREE_PATH = './resources/BiGCN/Twitter15/data.TD_RvNN.vol_5000.txt'
"""

"""
# Twitter 16
RAW_LABEL_PATH = './data/raw/rumor_detection_acl2017/twitter16/label.txt'
RAW_TREE_PATH = './data/raw/rumor_detection_acl2017/twitter16/tree/'
RESOURCE_LABEL_PATH = './resources/BiGCN/Twitter16/Twitter16_label_All.txt'
RESOURCE_TREE_PATH = './resources/BiGCN/Twitter16/data.TD_RvNN.vol_5000.txt'

# OUTPUT FILE
snapshot_num = 5
TIMESTAMPS_PATH = './data/sequence/Twitter16/timestamps.txt'

SEQUENCE_SNAPSHOT_PATH = './data/sequence/Twitter16/sequence_snapshot_{:02}.txt'.format(snapshot_num)
TEMPORAL_SNAPSHOT_PATH = './data/sequence/Twitter16/temporal_snapshot_{:02}.txt'.format(snapshot_num)
"""


# sequential


RAW_LABEL_PATH = ""
RAW_TREE_PATH = ""
RESOURCE_LABEL_PATH = ""
RESOURCE_TREE_PATH = ""

TIMESTAMPS_PATH = ""
SEQUENCE_SNAPSHOT_PATH = ""
TEMPORAL_SNAPSHOT_PATH = ""


TIMESTAMPS_PATH = './data/sequence/{}/timestamps.txt'.format(args['dataset_name'])
print('./data/sequence/{}/sequence_snapshot_{:02}.txt'.format(args['dataset_name'], int(args['snapshot_num'])))
print("HERE")
# TODO: sequence -> sequential
SEQUENCE_SNAPSHOT_PATH = './data/sequence/{}/sequence_snapshot_{:02}.txt'.format(
    args['dataset_name'], int(args['snapshot_num']))
TEMPORAL_SNAPSHOT_PATH = './data/sequence/{}/temporal_snapshot_{:02}.txt'.format(
    args['dataset_name'], int(args['snapshot_num']))


print("RAW_LABEL_PATH:", RAW_LABEL_PATH)
print("RAW_TREE_PATH:", RAW_TREE_PATH)
print("RESOURCE_LABEL_PATH:", RESOURCE_LABEL_PATH)
print("RESOURCE_TREE_PATH:", RESOURCE_TREE_PATH, end='\n\n')


def load_raw_labels():
    # RAW_LABEL_PATH: ./data/raw/rumor_detection_acl2017/twitter16/label.txt
    id_label_dict = {}
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    for line in open(RAW_LABEL_PATH):
        label, tweet_id = line.strip().split(":")
        id_label_dict[tweet_id] = label
        label_id_dict[label].append(tweet_id)
    print("{0} {1}".format(RAW_LABEL_PATH, len(id_label_dict)))
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_resource_labels():
    # RESOURCE_LABEL_PATH: ./resources/BiGCN/Twitter16/Twitter16_label_All.txt
    id_label_dict = {}
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    for line in open(RESOURCE_LABEL_PATH):
        elements = line.strip().split('\t')
        label, event_id = elements[0], elements[2]  # root_id
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    print("{0} {1}".format(RESOURCE_LABEL_PATH, len(id_label_dict)))
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_resource_trees():
    # RESOURCE_TREE_PATH: ./resources/BiGCN/Twitter16/data.TD_RvNN.vol_5000.txt
    trees_dict = {}
    for line in open(RESOURCE_TREE_PATH):
        elements = line.strip().split('\t')
        # event_id, parent_index, child_index = elements[0], elements[1], int(elements[2])
        # max_degree, max_post_len, word_features = int(elements[3]), int(elements[4]), elements[5]

        event_id = elements[0]
        parent_index = elements[1]
        child_index = int(elements[2])
        word_features = elements[5]

        if event_id not in trees_dict:
            trees_dict[event_id] = {}
        trees_dict[event_id][child_index] = {
            'parent_index': parent_index,
            'word_features': word_features,
        }

    print('trees count:', len(trees_dict), '\n')
    return trees_dict


def load_raw_trees():
    pass

# Load Temporal Information - Generate Sequence, Temporal Edge Index


def raw_tree_to_timestamps():
    temporal_info = {}
    for file_name in os.listdir(RAW_TREE_PATH):
        file_id = file_name[:-4]

        if file_id not in temporal_info:
            temporal_info[file_id] = []
        for index, line in enumerate(open(RAW_TREE_PATH + file_name)):
            elem_list = re.split(r"[\'\,\->\[\]]", line.strip())
            elem_list = [x.strip() for x in elem_list if x.strip()]
            src_user_id, src_tweet_id, src_time = elem_list[0:3]
            dst_user_id, dst_tweet_id, dst_time = elem_list[3:6]

            if elem_list[0] == 'ROOT' and elem_list[1] == 'ROOT':
                root_user_id, root_tweet_id = dst_user_id, dst_tweet_id

            elif src_tweet_id != dst_tweet_id:
                temporal_info[file_id].append(max(src_time, dst_time))

        temporal_info[file_id] = sorted(
            temporal_info[file_id], key=lambda x: float(x.strip()))

    save_json_file(TIMESTAMPS_PATH, temporal_info)
    # ensure_directory(TIMESTAMPS_PATH)
    # with open(TIMESTAMPS_PATH, "w") as sequence_file:
    #     sequence_file.write(json.dumps(temporal_info))

    return temporal_info


# def load_temporal_info():
#     with open(TIMESTAMPS_PATH, "r") as sequence_file:
#         temporal_info = json.loads(sequence_file.read())
#     return temporal_info


"""
def load_tree_length(TREE_PATH):
    tree_length_dict = {}
    for line in open(TREE_PATH):
        event_id = line.strip().split('\t')[0]
        if event_id not in tree_length_dict:
            tree_length_dict[event_id] = 0
        tree_length_dict[event_id] += 1
    return tree_length_dict
"""


def trim_temporal_info(resource_id_label_dict, temporal_info, resource_trees_dict):
    """
    match temporal information with resource dict
    """

    counter = [0, 0, 0, 0]
    for event_id in resource_id_label_dict:
        raw_temporal_length = len(temporal_info[event_id])
        resource_tree_length = len(resource_trees_dict[event_id]) - 1

        if raw_temporal_length > resource_tree_length:
            temporal_info[event_id] = temporal_info[event_id][:resource_tree_length]
            counter[0] += 1
        elif raw_temporal_length < resource_tree_length:
            counter[1] += 1
            diff_count = resource_tree_length - raw_temporal_length
            try:
                upsample = random.sample(temporal_info[event_id], diff_count)
            except:  # Unexpected
                counter[3] += 1
                if not len(temporal_info[event_id]):
                    upsample = ['10.0'] * diff_count
                else:
                    upsample = []
                    for _ in range(diff_count):
                        upsample.append(random.choice(temporal_info[event_id]))
            temporal_info[event_id] += upsample
            temporal_info[event_id] = sorted(
                temporal_info[event_id], key=lambda x: float(x.strip()))
        else:
            counter[2] += 1

    save_json_file(TIMESTAMPS_PATH, temporal_info)
    print(counter)

    return temporal_info


def sequence_to_snapshot_index(temporal_info):
    snapshot_edge_index = {}
    for event_id in temporal_info:
        if event_id not in snapshot_edge_index:
            snapshot_edge_index[event_id] = []
        sequence_len = len(temporal_info[event_id])
        base_edge_count = sequence_len % snapshot_num
        additional_edge_count = sequence_len // snapshot_num
        for snapshot_index in range(1, snapshot_num + 1):
            count = base_edge_count + additional_edge_count * snapshot_index
            snapshot_edge_index[event_id]
            snapshot_edge_index[event_id].append(count)

    save_json_file(SEQUENCE_SNAPSHOT_PATH, snapshot_edge_index)


def temporal_to_snapshot_index(temporal_info):
    snapshot_edge_index = {}
    for event_id in temporal_info:
        if event_id not in snapshot_edge_index:
            snapshot_edge_index[event_id] = []
        if not temporal_info[event_id]:
            snapshot_edge_index[event_id] = [0] * snapshot_num
            continue

        sequence = sorted(temporal_info[event_id],
                          key=lambda x: float(x.strip()))
        sequence = list(map(float, sequence))

        time_interval = (sequence[-1] - sequence[0]) / snapshot_num
        for snapshot_index in range(1, snapshot_num + 1):
            print('\nsnapshot_index', snapshot_index)
            edge_count = 0
            for seq in sequence:
                print(seq, end=' ')
                if seq <= time_interval * snapshot_index + sequence[0]:
                    edge_count += 1
                else:
                    break
            snapshot_edge_index[event_id].append(edge_count)

        snapshot_edge_index[event_id].pop()
        snapshot_edge_index[event_id].append(len(temporal_info[event_id]))  #

    save_json_file(TEMPORAL_SNAPSHOT_PATH, snapshot_edge_index)


# SEQUENCE DICT TO

# print(len(resource_trees_dict))
# print(resource_trees_dict[list(resource_trees_dict.keys())[0]])
# print(len(resource_trees_dict[list(resource_trees_dict.keys())[0]]))
# print(resource_trees_dict['553588178687655936'])
# print(len(resource_trees_dict['553588178687655936']))


def main():
    print("--------------------------------------")
    print("    RAW DATASET / RESOURCE DATASET    ")
    print("--------------------------------------")

    raw_id_label_dict, raw_label_id_dict = load_raw_labels()
    resource_id_label_dict, resource_label_id_dict = load_resource_labels()
    resource_trees_dict = load_resource_trees()
    temporal_info = raw_tree_to_timestamps()
    temporal_info = load_json_file(TIMESTAMPS_PATH)
    temporal_info = trim_temporal_info(
        resource_id_label_dict, temporal_info, resource_trees_dict)
    sequence_to_snapshot_index(temporal_info)
    temporal_to_snapshot_index(temporal_info)


if __name__ == '__main__':
    start_time = time.time()  # Timer Start
    main()
    end_time = time.time()
    print("\n")
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
