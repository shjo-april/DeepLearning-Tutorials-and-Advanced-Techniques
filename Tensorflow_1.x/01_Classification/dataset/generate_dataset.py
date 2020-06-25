import os
import cv2
import json

import sys

root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_dir)

from data.writer import *
from data.utils import *

from utility.utils import *

import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='', type=str)
    return parser.parse_args()

args = get_config()

data_dir = root_dir + '/dataset/{}/'.format(args.dataset_name)
sanghyun_dir = data_dir + 'sanghyun/'

tags = ['Train', 'Test']
json_paths = [data_dir + 'train.json', data_dir + 'test.json']

for tag, json_path in zip(tags, json_paths):
    data_dic = read_json(json_path)
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

