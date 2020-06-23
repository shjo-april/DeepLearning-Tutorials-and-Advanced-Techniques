import tensorflow as tf

from utility.utils import *
from utility.tensorflow_utils import *

train_dic = read_json('./dataset/train.json')

train_image_paths = list(train_dic.keys())
train_labels = [train_dic[image_path] for image_path in train_image_paths]

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

train_dataset = train_dataset.map(process_for_classification, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = prepare_for_training(train_dataset)

for image, label in iter(train_dataset):
    print(image.numpy().shape, label.numpy().shape)
    
