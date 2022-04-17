import os
import pathlib

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image

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
import cv2
"""
dataset_url = "C:/Users/Mihaela/Downloads/multiclass_trimmed"

data_dir = os.path.abspath(dataset_url)

data_dir = pathlib.Path(data_dir)
batch_size = 32
img_height = 227    #256
img_width = 227
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
"""
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
"""
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                                   input_shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
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
"""
"""data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),
    layers.Conv2D(32, 5, padding='same', activation='relu'),  # change kernel size to 5 from 3 for every conv layer
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 5, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 5, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 5, padding='same', activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    # layers.Dropout(0.5), #TRY CHANGE TO POINT 0.5 from 0.2
    layers.Flatten(),
    layers.Dense(512, activation='softmax'),  # CHANGED FROM relu and 128
    layers.Dense(num_classes)"""

"""model = Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),
    layers.Conv2D(96, 11, strides=4, activation='relu'),  # change kernel size to 5 from 3 for every conv layer
    #layers.Dropout(0.5),
    layers.MaxPooling2D(pool_size=3, strides=2),
    layers.Conv2D(256, 5, strides=1,padding='same', activation='relu'),
    #layers.Dropout(0.5),
    layers.MaxPooling2D(pool_size=3,strides=2),
    layers.Conv2D(384, 3, padding='same',strides=1, activation='relu'),
    #layers.Dropout(0.5),
    #layers.MaxPooling2D(),
    layers.Conv2D(384, 3, padding='same',strides=1, activation='relu'),
    layers.Conv2D(256, 3, padding='same',strides=1, activation='relu'),
    layers.MaxPooling2D(pool_size=3, strides=2),
    layers.Dropout(0.5), #TRY CHANGE TO POINT 0.5 from 0.2
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),  # CHANGED FROM relu and 128
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

print("Loading weights.")
model.load_weights('alexnet_layers_50_no_stop_new_aug.h5')
print("weights loaded!")

img_path_dir = 'C:\\Users\\Mihaela\\Downloads\\test_photo'

images = os.listdir(img_path_dir)

def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def pillow_brightness(img, brightness):
    bright = PIL.ImageEnhance.Brightness(img)
    img = bright.enhance(brightness)

    contrast = PIL.ImageEnhance.Contrast(img)
    return contrast.enhance(brightness)

def brightness_autocontrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    beta = -50
    alpha = beta / -minimum_gray
    # Calculate alpha and beta values
    # alpha = 255 / (maximum_gray - minimum_gray)
    # beta = -minimum_gray * alpha

    print(alpha, beta)

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return auto_result

for image_name in images:
    img_path = img_path_dir + "\\" + image_name
    img = image.load_img(img_path, target_size=(227, 227))  # for specific image size
    img_b = img.copy()
    for idx in range(70):
        # img = image.load_img(img_path)

        x = image.img_to_array(img_b)
        x = np.array(x)

        x = np.expand_dims(x, axis=0)
        #x = x * 1. / 256  # rescaled to same size as training
        #img = tf.keras.utils.load_img(
        #    img_path, target_size=(img_height, img_width)
        #)

        prediction = model.predict(x)


        score = tf.nn.softmax(prediction[0])
        class_names = ['correct', 'overexposed', 'underexposed']

        print(
            "Image {} most likely belongs to {} with a {:.2f} percent confidence."
                .format(image_name, class_names[np.argmax(score)], 100 * np.max(score))
        )

        if idx == 0 or class_names[np.argmax(score)] == 'correct':
            cv2.imshow("bob", cv2.cvtColor(np.uint8(image.img_to_array(img_b)), cv2.COLOR_BGR2RGB))
            key = cv2.waitKey()
            if key == ord('q'):
                exit()
            elif key == ord('x'):
                break

        if class_names[np.argmax(score)] == 'correct':
            break
        # x_b = gamma_trans(x_b, 1.05)
        # x_b = brightness_autocontrast(x_b)
        img_b = pillow_brightness(img_b, 0.99 if (class_names[np.argmax(score)] == 'overexposed') else 1.01)

