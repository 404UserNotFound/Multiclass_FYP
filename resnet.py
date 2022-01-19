import os

import tensorflow as tf
from flatbuffers.builder import np
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pathlib

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

dataset_url = "C:/Users/Mihaela/Downloads/multiclass_small"

data_dir = os.path.abspath(dataset_url)

data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir)
class_names = train_ds.class_names
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(data_dir, batch_size=20, class_mode='binary',
                                                    target_size=(224, 224))

validation_generator = test_datagen.flow_from_directory(data_dir, batch_size=20, class_mode='binary',
                                                        target_size=(224, 224))
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(1, activation='sigmoid'))
base_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
resnet_history = base_model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=100, epochs=10)
test_url = "C:\\Users\\Mihaela\\Downloads\\test_photo\\a0282-20060619_125715__MG_9197_0.JPG"

img = tf.keras.utils.load_img(
    test_url, target_size=(224, 224)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = base_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



