import glob

import numpy as np
import multiprocessing as mp

from data.reader import *
from data.utils import *

if __name__ == '__main__':
    # Loader
    queue = mp.Queue(5)
    training = False

    sanghyun_paths = glob.glob('./dataset/sanghyun/Train*')
    the_size_of_loading_files = 5

    loader = Loader(queue, training, sanghyun_paths, the_size_of_loading_files)
    iterator = Customized_Iterator(queue)

    # Decoder
    decoder_queue = mp.Queue(5)

    def decoder_func(example):
        image = example['encoded_image']
        label = example['label']

        image = decode_image(image)

        return image, label

    decoder = Decoder(decoder_queue, iterator, decoder_func)

    loader.start()
    decoder.start()

    i = 0
    
    while True:
        data = decoder_queue.get()
        if data == StopIteration:
            break
        
        image, label = data

        i += 1
        if i % 100 == 0:
            print('step1', i + 1, i % 100 == 0, image.shape, label, loader.is_alive())

    loader.close()
    decoder.close()

    print(loader.is_alive())
    input()

    loader = Loader(queue, training, sanghyun_paths, the_size_of_loading_files)
    iterator = Customized_Iterator(queue)

    decoder = Decoder(decoder_queue, iterator, decoder_func)
    
    loader.start()
    decoder.start()

    i = 0

    while True:
        data = decoder_queue.get()
        if data == StopIteration:
            break
        
        image, label = data

        i += 1
        if i % 100 == 0:
            print('step2', i + 1, i % 100 == 0, image.shape, label)
    
    loader.close()
    decoder.close()