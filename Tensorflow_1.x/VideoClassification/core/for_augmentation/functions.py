# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from core.for_augmentation.utils import *

class AvgPool2d():
    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure
        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)

class Random_HorizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return hflip(x)
        return x

class Random_VerticalFlip:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return vflip(x)
        return x

class Random_Crop:
    def __init__(self, crop_size):
        self.crop_w, self.crop_h = crop_size

    def __call__(self, x):
        h, w, c = x.shape
        
        xmin = random.randint(0, w - self.crop_w)
        ymin = random.randint(0, h - self.crop_h)

        return x[ymin : ymin + self.crop_h, xmin : xmin + self.crop_w, :]

class Random_Crop_with_Black:
    def __init__(self, crop_size):
        self.crop_w, self.crop_h = crop_size

    def __call__(self, x):
        h, w, c = x.shape

        crop_w = min(self.crop_w, w)
        crop_h = min(self.crop_h, h)

        space_w = w - self.crop_w
        space_h = h - self.crop_h

        if space_w > 0:
            left = 0
            x_left = random.randrange(space_w + 1)
        else:
            left = random.randrange(-space_w + 1)
            x_left = 0

        if space_h > 0:
            top = 0
            x_top = random.randrange(space_h + 1)
        else:
            top = random.randrange(-space_h + 1)
            x_top = 0

        image = np.zeros((self.crop_h, self.crop_w, c), dtype = np.uint8)
        image[top:top+crop_h, left:left+crop_w] = x[x_top:x_top+crop_h, x_left:x_left+crop_w]

        return image

class Padding:
    def __init__(self, size = 4):
        self.size = size

    def __call__(self, x):
        return add_padding(x, self.size)

class Random_ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x):
        transforms = []

        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(lambda img: adjust_brightness(img, brightness_factor))

        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(lambda img: adjust_contrast(img, contrast_factor))

        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(lambda img: adjust_saturation(img, saturation_factor))

        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            transforms.append(lambda img: adjust_hue(img, hue_factor))

        random.shuffle(transforms)

        for transform in transforms:
            x = transform(x)

        return x

class Center_Crop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        return center_crop(x, self.crop_size)

class Resize:
    def __init__(self, image_size):
        self.interpolation_mode = {
            'NEAREST' : cv2.INTER_NEAREST,
            'BILINEAR' : cv2.INTER_LINEAR,
            'BICUBIC' : cv2.INTER_CUBIC,
        }
        
        self.image_size = image_size
        self.interpolation_names = list(self.interpolation_mode.keys())
    
    def __call__(self, image, name = None):
        if image.shape[:2] == self.image_size:
            return image

        if name is None:
            name = random.choice(self.interpolation_names)

        return cv2.resize(image, self.image_size, self.interpolation_mode[name])

class Random_Resize:
    def __init__(self, min_image_size, max_image_size):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size

    def __call__(self, image):
        h, w, c = image.shape
        image_size = random.randint(self.min_image_size, self.max_image_size)

        if w < h:
            w, h = round(w * image_size / h), image_size
        else:
            w, h = image_size, round(h * image_size / w)

        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)
        return image

class Fixed_Resize:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        try:
            h, w, c = image.shape
        except ValueError:
            h, w = image.shape

        if w > self.image_size and h > self.image_size:
            if w < h:
                w, h = round(w * self.image_size / h), self.image_size
            else:
                w, h = self.image_size, round(h * self.image_size / w)
            
            image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)
            
        return image

class Top_Left_Crop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        return top_left_crop(image, self.crop_size, 0)

