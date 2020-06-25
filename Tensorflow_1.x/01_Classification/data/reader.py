import numpy as np
import multiprocessing as mp

from data import utils

class Customized_Iterator:
    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        data = self.queue.get()
        if data == StopIteration:
            raise StopIteration
        return data

class Loader(mp.Process):
    def __init__(self, queue, training, sanghyun_paths, the_number_of_decoder, the_size_of_loading_files):
        super().__init__()

        self.queue = queue
        self.training = training

        self.sanghyun_paths = sanghyun_paths

        self.the_number_of_decoder = the_number_of_decoder
        self.the_size_of_loading_files = the_size_of_loading_files

    def run(self):
        separated_paths = utils.customized_split_using_size(self.sanghyun_paths, self.the_size_of_loading_files, shuffle=self.training)
        for paths in separated_paths:
            
            dataset = {}
            for path in paths:
                dataset.update(utils.load_pickle(path))

            keys = list(dataset.keys())
            if self.training:
                np.random.shuffle(keys)

            for key in keys:
                self.queue.put(dataset[key])

        for _ in range(self.the_number_of_decoder):
            self.queue.put(StopIteration)

    def close(self):
        self.terminate()
        self.join()

class Decoder(mp.Process):
    def __init__(self, queue, iterator, decode_func):
        super().__init__()

        self.queue = queue
        self.iterator = iterator

        self.decode_func = decode_func

    def run(self):
        for example in self.iterator:
            self.queue.put(self.decode_func(example))
        self.queue.put(StopIteration)

    def close(self):
        self.terminate()
        self.join()

class Sanghyun_Reader:

    def __init__(self, 
    
                 sanghyun_paths, 
                 training, 
                 drop_remainder, 
                 batch_size, 
                 image_shape, 
                 
                 the_number_of_loader, 
                 the_number_of_decoder, 
                 
                 decode_fn,
                 names
        ):

        self.training = training
        self.drop_remainder = drop_remainder

        self.sanghyun_paths = sanghyun_paths

        self.batch_size = batch_size
        self.image_height, self.image_width, self.image_channel = image_shape

        self.the_size_of_loading_files = min(5, len(sanghyun_paths))
        
        self.queue_of_loader = mp.Queue(maxsize=2)
        self.queue_of_decoder = mp.Queue(maxsize=self.batch_size)

        self.iterator_of_loader = Customized_Iterator(self.queue_of_loader)
        self.iterator_of_decoder = Customized_Iterator(self.queue_of_decoder)

        self.decode_fn = decode_fn

        self.the_number_of_loader = the_number_of_loader
        self.the_number_of_decoder = the_number_of_decoder

        self.loader_option = {
            'queue' : self.queue_of_loader,
            'training' : self.training,
            'sanghyun_paths' : self.sanghyun_paths,

            'the_number_of_decoder' : self.the_number_of_decoder,
            'the_size_of_loading_files' : self.the_size_of_loading_files
        }

        self.decoder_option = {
            'queue' : self.queue_of_decoder,
            'iterator' : self.iterator_of_loader,
            'decode_func' : self.decode_fn,
        }

        self.names = names
        self.init_dataset()

    def start(self):
        self.loaders = [Loader(**self.loader_option) for _ in range(self.the_number_of_loader)]
        self.decoders = [Decoder(**self.decoder_option) for _ in range(self.the_number_of_decoder)]

        for obj in self.loaders:
            obj.start()

        for obj in self.decoders:
            obj.start()

    def close(self):
        for obj in self.loaders:
            obj.close()

        for obj in self.decoders:
            obj.close()

    def init_dataset(self):
        self.dataset = {name : [] for name in self.names}
        self.dataset['length'] = 0

    def get_dataset(self):
        dataset = [np.asarray(self.dataset[name], dtype=np.float32) for name in self.names]
        self.init_dataset()
        return dataset
    
    def __iter__(self):
        return self

    def __next__(self):
        for data in self.iterator_of_decoder:
            for name, d in zip(self.names, data):
                self.dataset[name].append(d)

            self.dataset['length'] += 1
            if self.dataset['length'] == self.batch_size:
                return self.get_dataset()
            
        if not self.drop_remainder and self.dataset['length'] > 0:
            return self.get_dataset()

        raise StopIteration
        