import os
import glob

import sys

root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utility.utils import read_json, write_json

data_dir = '../../DB/flower_photos/'
json_dir = root_dir + '/dataset/flower_photos/'

if not os.path.isdir(json_dir):
    os.makedirs(json_dir)

class_names = sorted(os.listdir(data_dir)); print(class_names)

train_dataset = {}
test_dataset = {}

for label, class_name in enumerate(class_names):
    image_paths = glob.glob(data_dir + class_name + '/*')
    length = len(image_paths)

    train_length = int(length * 0.8)

    for path in image_paths[:train_length]:
        train_dataset[path] = label

    for path in image_paths[train_length:]:
        test_dataset[path] = label

write_json(json_dir + 'train.json', train_dataset)
write_json(json_dir + 'test.json', test_dataset)

