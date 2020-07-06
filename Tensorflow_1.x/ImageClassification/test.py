
import cv2

import numpy as np
import tensorflow as tf

from utility.utils import *
from core.for_augmentation.functions import *

class Model:
    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(graph_def, name = 'prefix')

class Classifier(Model):
    def __init__(self, dataset, gpu_usage = 0.10, detect_threshold = 0.5):
        self.load_graph('./experiments/{}/pb/model.pb'.format(dataset))

        self.image_var = self.graph.get_tensor_by_name('prefix/images:0')
        self.predictions_op = self.graph.get_tensor_by_name('prefix/Classifier/predictions:0')

        shape = self.image_var.shape.as_list()
        _, self.height, self.width, _ = shape

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
        config.gpu_options.allow_growth = True

        self.detect_threshold = detect_threshold
        self.sess = tf.Session(graph = self.graph, config = config)
        
        self.image_size = self.height
        
        self.resize_func = Fixed_Resize(self.image_size)
        self.crop_func = Top_Left_Crop((self.image_size, self.image_size))
        
        self.class_names = [name.strip() for name in open('./dataset/{}/class_names.txt'.format(dataset)).readlines()]
        self.class_names = np.asarray(self.class_names)
    
    def predict(self, image):
        h, w, c = image.shape

        image = self.resize_func(image)
        image = self.crop_func(image)

        class_prob = self.sess.run(self.predictions_op, feed_dict = {self.image_var : [image]})[0]

        class_index = np.argmax(class_prob)
        class_prob = class_prob[class_index]

        return self.class_names[class_index], class_prob

if __name__ == '__main__':
    def get_config():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', default='', type=str)
        return parser.parse_args()

    config = get_config()

    model = Classifier(config.dataset_name)

    data_dic = read_json('./dataset/{}/test.json'.format(config.dataset_name))
    
    for image_path in sorted(data_dic.keys()):
        image = cv2.imread(image_path)
        label = data_dic[image_path]

        class_name, class_prob = model.predict(image)

        print('# {}, GT = {}, Prediction {} = {:.2f}%'.format(image_path, model.class_names[label], class_name, class_prob * 100))

