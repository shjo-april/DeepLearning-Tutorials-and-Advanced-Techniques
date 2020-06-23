
import glob

import numpy as np
import multiprocessing as mp

from core.augmentors import *

from data.reader import *
from data.utils import *

from utility.utils import *
from utility.timer import *

if __name__ == '__main__':
    image_size = 299
    max_image_size = int(image_size * 1.25)
    min_image_size = max_image_size // 2

    train_transforms = DataAugmentation(
        [
            Random_Resize(min_image_size, max_image_size),
            # RandAugment(),
            Random_Crop_with_Black((image_size, image_size))
        ]
    )
    
    timer = Timer()
    timer.tik()
    
    # tags = ['Train', 'Test']
    # json_paths = ['./dataset/train.json', './dataset/test.json']

    tags = ['Train']
    json_paths = ['./dataset/train.json']
    
    for tag, json_path in zip(tags, json_paths):
        data_dic = read_json(json_path)
        image_paths = list(data_dic.keys())

        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = train_transforms(image)

    # with RandAugment = 11.183sec
    # without RandAugment = 7.553sec
    print('{}ms'.format(timer.tok(ms=True)))

