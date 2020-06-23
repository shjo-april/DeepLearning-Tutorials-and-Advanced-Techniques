
import glob

import numpy as np
import multiprocessing as mp

from core.augmentors import *

from data.reader import *
from data.utils import *

from utility.timer import *

if __name__ == '__main__':
    image_size = 299
    max_image_size = int(image_size * 1.25)
    min_image_size = max_image_size // 2

    train_transforms = DataAugmentation(
        [
            Random_Resize(min_image_size, max_image_size),
            RandAugment(),
            Random_Crop_with_Black((image_size, image_size))
        ]
    )

    def decoder_func(example):
        image = example['encoded_image']
        label = example['label']

        image = decode_image(image)
        image = train_transforms(image)

        return image, label
    
    train_reader_option = {
        'sanghyun_paths' : glob.glob('./dataset/sanghyun/Train*'),
        'training' : False,
        'drop_remainder' : False,
        'batch_size' : 64,
        'image_shape' : (224, 224, 3),
        
        'the_number_of_loader' : 1, 
        'the_number_of_decoder' : 12, 
        
        'decode_fn' : decoder_func,
        'names' : ['image', 'label']
    }
    train_reader = Sanghyun_Reader(**train_reader_option)

    timer = Timer()
    timer.tik()

    train_reader.start()

    size = 0
    for images, labels in train_reader:
        size += images.shape[0]

    print('close')
    train_reader.close()

    # with RandAugment = 2.015sec
    # without RandAugment = 1.412sec
    print('{}ms'.format(timer.tok(ms=True)))

