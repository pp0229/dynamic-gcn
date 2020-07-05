
import os
import json

"""
    Author: Jiho Choi (jihochoi@snu.ac.kr)
    Usage:
        - sys.path.insert(0, './[parent-directory]/')
"""


def ensure_directory(path):
    path = os.path.split(path)
    if not os.path.exists(path[0]):
        os.makedirs(path[0])


def save_json_file(path, data):
    ensure_directory(path)
    with open(path, "w") as json_file:
        json_file.write(json.dumps(data))


def load_json_file(path):
    with open(path, "r") as json_file:
        data = json.loads(json_file.read())
    return data


def print_dict(dict_file):
    for key in dict_file.keys():
        print("\t {0}: {1}".format(key, dict_file[key]))

