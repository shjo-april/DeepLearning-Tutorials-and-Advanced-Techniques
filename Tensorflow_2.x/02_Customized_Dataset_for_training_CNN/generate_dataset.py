import os
import glob

from utility.utils import read_json, write_json

root_dir = '../DB/flower_photos/'
class_names = sorted(os.listdir(root_dir)); print(class_names)

train_dataset = {}
test_dataset = {}

for label, class_name in enumerate(class_names):
    image_paths = glob.glob(root_dir + class_name + '/*')
    length = len(image_paths)

    train_length = int(length * 0.8)

    for path in image_paths[:train_length]:
        train_dataset[path] = label

    for path in image_paths[train_length:]:
        test_dataset[path] = label

write_json('./dataset/train.json', train_dataset)
write_json('./dataset/test.json', test_dataset)

