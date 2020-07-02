import os
import glob

import sys

root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_dir)

from data.writer import *
from data.utils import *

from utility.utils import *

def get_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='', type=str)
    return parser.parse_args()

args = get_config()

data_dir = '../../DB/' + args.dataset_name + '/'
json_dir = root_dir + '/dataset/' + args.dataset_name + '/'
sanghyun_dir = json_dir + 'sanghyun/'

if not os.path.isdir(json_dir):
    os.makedirs(json_dir)

if not os.path.isdir(sanghyun_dir):
    os.makedirs(sanghyun_dir)

class_names = sorted(os.listdir(data_dir))

with open(json_dir + 'class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')

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

# print(train_dataset)
# print(test_dataset)

write_json(json_dir + 'train.json', train_dataset)
write_json(json_dir + 'test.json', test_dataset)

tags = ['Train', 'Test']
datasets = [train_dataset, test_dataset]

for tag, data_dic in zip(tags, datasets):
    image_paths = list(data_dic.keys())
    
    writer = Sanghyun_Writer(sanghyun_dir, tag + '_{}', 100)
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        label = data_dic[image_path]
        
        encoded_image = encode_image(image)
        
        key = os.path.basename(image_path)
        example = {
            'image_path' : image_path,
            'encoded_image' : encoded_image,
            'label' : label
        }
        writer(key, example)
    
    writer.save()