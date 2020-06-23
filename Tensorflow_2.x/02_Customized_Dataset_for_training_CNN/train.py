import os

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
            Random_HorizontalFlip(),
            Random_Crop_with_Black((224, 224))
        ]
    ),
    'shuffle' : True,
}
train_ds = Dataset_for_classification(**train_option)

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
    ),
    'shuffle' : False,
}
test_ds = Dataset_for_classification(**test_option)

# model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

gap_layer = tf.keras.layers.GlobalAveragePooling2D()
logits_layer = tf.keras.layers.Dense(5)

model = tf.keras.Sequential([
    base_model,
    gap_layer,
    logits_layer
])

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.016, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)

        loss = compute_loss(labels, logits)
        accuracy = compute_accuracy(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, accuracy

def train(model, optimizer, dataset, log_freq=50):
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_accuracy = tf.keras.metrics.Mean('accuracy', dtype=tf.float32)

    for images, labels in dataset:
        loss, accuracy = train_step(model, optimizer, images, labels)

        avg_loss(loss)
        avg_accuracy(accuracy)

        if tf.equal(optimizer.iterations % log_freq, 0):
            print('# step:', int(optimizer.iterations), ', train_loss:', avg_loss.result().numpy(), ', train_accuracy:', avg_accuracy.result().numpy())

            avg_loss.reset_states()
            avg_accuracy.reset_states()

def test(model, dataset):
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_accuracy = tf.keras.metrics.Mean('accuracy', dtype=tf.float32)

    for (images, labels) in dataset:
        logits = model(images, training=False)

        avg_loss(compute_loss(labels, logits))
        avg_accuracy(compute_accuracy(labels, logits))

    print('# test_loss :', avg_loss.result().numpy(), ', test accurary :', avg_accuracy.result().numpy())

root_dir = './experiments/'
checkpoint_dir = root_dir + 'checkpoint/'
export_path = root_dir + 'export/'

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for i in range(100):
    train(model, optimizer, train_ds, log_freq=10)
    test(model, test_ds)

    checkpoint.save(checkpoint_prefix)

tf.saved_model.save(model, export_path)

