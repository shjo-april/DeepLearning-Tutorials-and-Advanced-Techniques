import os

from core.model import *
from core.dataset import *
from core.augmentors import *

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

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.016, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

def train(model, optimizer, dataset, log_freq=50):
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_accuracy = tf.keras.metrics.Mean('accuracy', dtype=tf.float32)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)

            loss = loss_fn(labels, logits)
            accuracy = accuracy_fn(labels, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss, accuracy
    
    for images, labels in dataset:
        loss, accuracy = train_step(images, labels)
        
        avg_loss(loss)
        avg_accuracy(accuracy)

        if tf.equal(optimizer.iterations % log_freq, 0):
            print('# step:', int(optimizer.iterations), ', train_loss:', avg_loss.result().numpy(), ', train_accuracy:', avg_accuracy.result().numpy())

            avg_loss.reset_states()
            avg_accuracy.reset_states()

def test(model, dataset):
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_accuracy = tf.keras.metrics.Mean('accuracy', dtype=tf.float32)

    for images, labels in dataset:
        logits = model(images, training=False)

        avg_loss(loss_fn(labels, logits))
        avg_accuracy(accuracy_fn(labels, logits))

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
