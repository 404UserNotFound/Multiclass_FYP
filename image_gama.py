from datetime import datetime
from keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import cv2

# Mihaela Brodetchi C00242687

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                           input_shape=(227, 227, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Load in weights from previously trained model
model.load_weights('alexnet_layers_50_no_stop_new_aug.h5')
print("Weights loaded!")

# Get image path of testing dataset
img_path_dir = os.getcwd()
img_path_dir = img_path_dir + '\\test_photo'
images = os.listdir(img_path_dir)


# Image adjustment by gamma transformation
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


# Loop through the test dataset
for image_name in images:
    img_path = img_path_dir + "\\" + image_name
    img = image.load_img(img_path, target_size=(227, 227))
    img_large = image.load_img(img_path)
    img_b = img.copy()
    x = image.img_to_array(img_b)
    x_b = np.array(x)
    for idx in range(70):
        x = np.expand_dims(x_b, axis=0)
        # make prediction
        prediction = model.predict(x)
        score = tf.nn.softmax(prediction[0])
        class_names = ['correct', 'overexposed', 'underexposed']
        print(
            "Input {} most likely belongs to {} with a {:.2f} percent confidence."
                .format(image_name, class_names[np.argmax(score)], 100 * np.max(score))
        )
        # display the image
        if idx == 0 or class_names[np.argmax(score)] == 'correct':
            cv2.imshow("Image", cv2.cvtColor(np.uint8(image.img_to_array(x_b)), cv2.COLOR_BGR2RGB))
            key = cv2.waitKey()
            # q to quit
            if key == ord('q'):
                exit()
            # x to skip image
            elif key == ord('x'):
                break
        # if image class is correct, get date time as string and save image with timestamp as filename
        if class_names[np.argmax(score)] == 'correct':
            date_time = datetime.now()
            date_time = date_time.strftime("%m%d%H%M%S%f")
            date_str = date_time + ".jpg"
            cv2.imwrite(date_str, img_large)
            break
        # image x_b is used on the display screen and to check exposure with neural network
        # img_large is edited separately. This is because the input image for the neural network must be 227*227
        # but the image saved to disk should be full sized.
        x_b = gamma_trans(np.uint8(x_b), 1.05)
        img_large = gamma_trans(cv2.cvtColor(np.uint8(img_large), cv2.COLOR_BGR2RGB), 1.05)