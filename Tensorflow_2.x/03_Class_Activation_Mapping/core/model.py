import numpy as np
import tensorflow as tf

class Inception_ResNet_v2_with_CAM(tf.keras.Model):
    def __init__(self, image_shape, classes):
        super(Inception_ResNet_v2_with_CAM, self).__init__()

        self.backbone = tf.keras.applications.InceptionResNetV2(
            input_shape=image_shape,
            include_top=False,
            weights='imagenet'
        )

        self.fc = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            use_bias=False
        )
        
        self.relu = tf.keras.layers.Activation('relu')
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=(8, 8),
        )
        self.flatten = tf.keras.layers.Flatten()

    def call(self, images, training=None, using_cam=False):

        x = self.backbone(images)

        if using_cam:
            x = self.fc(x)
            x = self.relu(x)
            
        else:
            x = self.pool(x)
            x = self.fc(x)
            x = self.flatten(x)

        return x

if __name__ == '__main__':

    # (1, 8, 8, 1536)
    x = np.zeros((1, 299, 299, 3))
    model = Inception_ResNet_v2_with_CAM((299, 299, 3), 5)

    logits = model(x, training=True, using_cam=False)
    cams = model(x, training=True, using_cam=True)

    logits_from_cams = tf.math.reduce_mean(cams, axis=(1, 2))
    
    print(logits)
    print(logits_from_cams)    

    print(logits.shape, logits == logits_from_cams)
    print(cams.shape)