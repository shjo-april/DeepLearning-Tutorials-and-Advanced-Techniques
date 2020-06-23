
from data import utils

class Sanghyun_Writer:
    def __init__(self, dataset_dir, dataset_name_format, the_number_of_example):
        self.dataset_index = 1
        self.dataset_format = dataset_dir + dataset_name_format + '.sanghyun'

        self.accumulated_size = 0
        self.the_number_of_example = the_number_of_example
        
        self.dataset = {}
    
    def __call__(self, key, example):    
        self.accumulated_size += 1
        self.dataset[key] = example

        if self.accumulated_size == self.the_number_of_example:
            self.save()

            self.accumulated_size = 0
            self.dataset = {}
    
    def get_path(self):
        path = self.dataset_format.format(self.dataset_index)
        return path

    def save(self):
        if self.accumulated_size > 0:
            print('# Write path : {}'.format(self.get_path()))
            utils.dump_pickle(self.get_path(), self.dataset)
            self.dataset_index += 1
