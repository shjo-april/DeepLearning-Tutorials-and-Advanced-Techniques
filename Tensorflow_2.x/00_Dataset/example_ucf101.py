
import tensorflow_datasets as tfds

# print(tfds.list_builders())

ds = tfds.load('ucf101', split='train', shuffle_files=False)
print(ds)

