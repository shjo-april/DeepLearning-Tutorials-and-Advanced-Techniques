import os

from core.model import *
from core.dataset import *
from core.augmentors import *

from utility.tensorflow_utils import *

# 1. Dataset
image_size = 299
batch_size = 16
classes = 5

max_image_size = int(image_size * 1.25)
min_image_size = max_image_size // 2

train_transforms = DataAugmentation(
        [
            Random_Resize(min_image_size, max_image_size),
            Random_HorizontalFlip(),
            Random_Crop_with_Black((image_size, image_size))
        ]
)

test_transforms = DataAugmentation(
    [
        Fixed_Resize(image_size),
        Top_Left_Crop((image_size, image_size))
    ]
)

dataset_option = {
    'train_json_path' : './dataset/train.json',
    'test_json_path' : './dataset/test.json',

    'train_transforms' : train_transforms,
    'test_transforms' : test_transforms,

    'image_size' : [image_size, image_size],
    'batch_size' : batch_size,
    'classes' : classes
}
train_ds, test_ds = create_datasets(**dataset_option)

# 2. Model
model = Inception_ResNet_v2_with_CAM((image_size, image_size, 3), classes)

def test(model, dataset):
    for images, labels in dataset:
        cams = model(images, training=False, using_cam=True)
        
        cams = tf.image.resize(cams, (image_size, image_size), method='bilinear')
        cams = normalize(cams) * 255
        
        for image, cam in zip(images, cams.numpy()):
            for class_index in range(classes):
                cv2.imshow('cam - {}'.format(class_index), cam[..., class_index].astype(np.uint8))
            
            cv2.imshow('show', image.astype(np.uint8))
            cv2.waitKey(0)

root_dir = './experiments/'
checkpoint_dir = root_dir + 'checkpoint/'
export_path = root_dir + 'export/'

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_ds:
    print(images.shape, labels.shape)

test(model, test_ds)
tf.saved_model.save(model, export_path)

