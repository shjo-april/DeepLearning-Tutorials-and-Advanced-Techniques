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
    video_paths = glob.glob(data_dir + class_name + '/*')
    length = len(video_paths)

    train_length = int(length * 0.8)

    for path in video_paths[:train_length]:
        train_dataset[path] = label

    for path in video_paths[train_length:]:
        test_dataset[path] = label

# print(train_dataset)
# print(test_dataset)

write_json(json_dir + 'train.json', train_dataset)
write_json(json_dir + 'test.json', test_dataset)

tags = ['Train', 'Test']
datasets = [train_dataset, test_dataset]

for tag, data_dic in zip(tags, datasets):
    video_paths = list(data_dic.keys())
    
    writer = Sanghyun_Writer(sanghyun_dir, tag + '_{}', 50)
    
    for video_path in video_paths:
        video = cv2.VideoCapture(video_path)
        label = data_dic[video_path]

        fps = video.get(cv2.CAP_PROP_FPS)

        encoded_frames = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break

            encoded_frame = encode_image(frame)
            encoded_frames.append(encoded_frame)

            if len(encoded_frames) >= fps * 4:
                break
        
        key = os.path.basename(video_path)
        example = {
            'video_path' : video_path,
            'encoded_frames' : encoded_frames,
            'label' : label
        }
        writer(key, example)
    
    writer.save()

