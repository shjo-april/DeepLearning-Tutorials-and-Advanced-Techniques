
from core.dataset import *
from core.augmentors import *

max_image_size = int(224 * 1.25)
min_image_size = max_image_size // 2

train_option = {
    'json_path' : './dataset/train.json',
    'image_size' : (224, 224),
    'batch_size' : 64,
    'classes' : 5,
    'transforms' : DataAugmentation(
        [
            Random_Resize(min_image_size, max_image_size),
            RandAugment(),
            Random_Crop_with_Black((224, 224))
        ]
    )
}
train_dataset = Dataset_for_classification(**train_option)

test_option = {
    'json_path' : './dataset/test.json',
    'image_size' : (224, 224),
    'batch_size' : 64,
    'classes' : 5,
    'transforms' : DataAugmentation(
        [
            Fixed_Resize(224),
            Top_Left_Crop((224, 224))
        ]
    )
}
test_dataset = Dataset_for_classification(**test_option)

for images, labels in train_dataset:
    print(images.shape, labels.shape)

