import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

gap_layer = tf.keras.layers.GlobalAveragePooling2D()
logits_layer = tf.keras.layers.Dense(5)

base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    gap_layer,
    logits_layer
])

print('The number of layers : {}'.format(len(model.layers)))

model.summary()

