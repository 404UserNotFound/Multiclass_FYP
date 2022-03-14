import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import pathlib

dataset_url = "C:/Users/Mihaela/Downloads/multiclass"

data_dir = os.path.abspath(dataset_url)

data_dir = pathlib.Path(data_dir)
batch_size = 32
img_height = 256
img_width = 256
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

"""model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)

])"""
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                                   input_shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)
# VISUALISE DATA AUG
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

model = Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    # layers.Dropout(0.5), #TRY CHANGE TO POINT 0.5 from 0.2
    layers.Flatten(),
    layers.Dense(512, activation='relu'),  # CHANGED FROM relu and 128
    layers.Dense(num_classes)
])

print("Loading weights.")
model.load_weights('relu_fullimgsize_removeLRplateau_512dense.h5')
print("weights loaded!")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
#learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1)
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300,
    callbacks=[early_stop]  # add learning_rate too
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
test_url = "C:\\Users\\Mihaela\\Downloads\\test_photo\\barely_under.JPG"

img = tf.keras.utils.load_img(
    test_url, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
# serialize weights to HDF5
model.save_weights('relu_fullimgsize_removeLRplateau_512dense2.h5')
# model.save('./model_tf', save_format='tf') THIS DOES NOT WORK WITH AUGMENTATION LAYERS
print("Saved model to disk")
