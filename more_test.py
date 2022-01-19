import os
import pathlib

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image

dataset_url = "C:/Users/Mihaela/Downloads/multiclass_small"

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

num_classes = len(class_names)

model = Sequential([
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
])

model.load_weights('weights.h5')


img_path = 'C:\\Users\\Mihaela\\Downloads\\test_photo\\a0228-IMG_2688.jpg'
img = image.load_img(img_path, target_size=(256, 256)) # if a you want a spesific image size
#img = image.load_img(img_path)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x*1./256 #rescale as training
img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
)

prediction = model.predict(x)
score = tf.nn.softmax(prediction[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)