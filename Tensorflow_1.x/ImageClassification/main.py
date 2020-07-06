
import os

def get_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--augmentation', default='default', type=str)
    return parser.parse_args()

config = get_config()

os.system('python3 ./dataset/generate_dataset.py --dataset_name {}'.format(config.dataset_name))
os.system('python3 train.py --dataset_name {} --augmentation {}'.format(config.dataset_name, config.augmentation))
os.system('python3 make_pb_and_tflite.py --dataset_name {}'.format(config.dataset_name))

