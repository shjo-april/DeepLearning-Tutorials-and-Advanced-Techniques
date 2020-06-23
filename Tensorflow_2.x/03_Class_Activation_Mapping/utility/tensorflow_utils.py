import tensorflow as tf

def normalize(cams):
    min_values = tf.math.reduce_min(cams, axis=(1, 2), keepdims=True)
    max_values = tf.math.reduce_max(cams, axis=(1, 2), keepdims=True)

    cams = (cams - min_values) / (max_values - min_values + 1e-5)
    return cams

def decode_image(image):
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.decode_png(image, channels=3))

    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, [224, 224])

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = decode_image(image)
    return image

def process_for_classification(image_path, label):
    image = load_image(image_path)
    return image, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(64)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds