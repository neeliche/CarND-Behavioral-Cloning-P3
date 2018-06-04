import csv
from random import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Convolution2D, ELU, Flatten, Cropping2D
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib

# Ensuring my devices are detected.
device_lib.list_local_devices()


# Trying out comma.ai model https://github.com/commaai/research/blob/master/train_steering_model.py
def get_comma_ai_model():
    row, col = 160, 320  # Per the lectures

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(row, col, 3)))
    model.add(Cropping2D(cropping=((70, 25), (40, 40)), input_shape=(3, 160, 320)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    return model


# Trying out the NVIDIA model:
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def get_nvidia_model():
    row, col = 160, 320  # Per the lectures

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(row, col, 3)))
    model.add(Cropping2D(cropping=((70, 25), (40, 40)), input_shape=(3, 160, 320)))

    '''
    Per the paper: first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution
    # with a 3×3 kernel size in the last two convolutional layers.
    '''
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="valid", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="valid", W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


# Switch to the model that is needed.
def get_model():
    return get_nvidia_model()
    # return get_comma_ai_model()


# Returns the file name from the image path. Some were linux generated (sample ones from Udacity) and others were
# generated using windows. So the path format changes.
def get_file_name(source_path):
    return source_path.split('/')[-1] if source_path.find('/') != -1 else source_path.split("\\")[-1]


# For NVIDIA model, pre-processing it into YUV color space as mentioned in the paper.
def preprocess_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Pre generator model where all the images are loaded first and then sent to the Model
def get_data(folder):
    lines = []
    imgs = []
    measures = []
    correction = 0.2
    with open('./' + folder + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    del lines[0]
    for line in lines:
        if float(line[6]) <= 0.0:
            continue

        image_center = preprocess_image(cv2.imread('./' + folder + '/IMG/' + get_file_name(line[0])))
        image_left = preprocess_image(cv2.imread('./' + folder + '/IMG/' + get_file_name(line[1])))
        image_right = preprocess_image(cv2.imread('./' + folder + '/IMG/' + get_file_name(line[2])))
        measurement_center = float(line[3])
        measurement_right = measurement_center - correction
        measurement_left = measurement_center + correction
        imgs.extend((image_center, image_left, image_right))
        measures.extend((measurement_center, measurement_left, measurement_right))
    return imgs, measures


# Gets only the image paths and the measurements that can be fed to the Generator and loaded dynamically as needed.
def get_image_paths(folder):
    lines = []
    imgs = []
    measures = []
    correction = 0.2
    with open('./' + folder + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    del lines[0]
    for line in lines:
        if float(line[6]) <= 0.0:
            continue

        image_center = './' + folder + '/IMG/' + get_file_name(line[0])
        image_left = './' + folder + '/IMG/' + get_file_name(line[1])
        image_right = './' + folder + '/IMG/' + get_file_name(line[2])
        measurement_center = float(line[3])
        measurement_right = measurement_center - correction
        measurement_left = measurement_center + correction
        imgs.extend((image_center, image_left, image_right))
        measures.extend((measurement_center, measurement_left, measurement_right))
    return imgs, measures


# A simple generator for dynamically getting the images as and when needed to improve the initial processing speed
def generator(image_paths, steering_angles, batch_size=128):
    # Does the flip augmentation so batch_size is actually double of what is sent

    while 1:
        aug_imgs, aug_measures = [], []
        for i in range(len(image_paths)):
            # print(i)
            img = preprocess_image(cv2.imread(image_paths[i]))
            measure = steering_angles[i]
            aug_imgs.append(img)
            aug_imgs.append(cv2.flip(img, 1))
            aug_measures.append(measure)
            aug_measures.append(measure * -1.0)

            if len(aug_imgs) == batch_size * 2:
                X_train = np.array(aug_imgs)
                y_train = np.array(aug_measures)
                aug_imgs, aug_measures = [], []
                yield shuffle(X_train, y_train)


'''
Pre Generator code. 
'''
# sample_images, sample_measurements = get_data('sample_data')
# train_images, train_measurements = get_data('Data')
# images = sample_images
# images.extend(train_images)
# measurements = sample_measurements
# measurements.extend(train_measurements)
#
# augmented_images, augmented_measurements = [], []
#
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_measurements.append(measurement)
#     augmented_measurements.append(measurement*-1.0)
#
# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)


'''
Load all the images
'''
sample_images, sample_measurements = get_image_paths('sample_data')
input_images, input_measurements = get_image_paths('Data')
images = sample_images
images.extend(input_images)
measurements = sample_measurements
measurements.extend(input_measurements)


'''
Split the data into training and validation sets 80/20 split
'''
training_images, validation_images, training_measurement, validation_measurement = \
    train_test_split(images, measurements, test_size=0.2)


'''
Generators for the Training and validation sets
'''
train_generator = generator(training_images, training_measurement)
valid_generator = generator(validation_images, validation_measurement)


'''
Get the model and perform the operation/training
'''
model = get_model()
history_object = model.fit_generator(train_generator, validation_data=valid_generator,
                                     nb_val_samples=2 * len(validation_images),
                                     nb_epoch=5, samples_per_epoch=2 * len(training_images), verbose=1)

print("Model Summary:")
print(model.summary())


'''
Plot the training and validation loss for each epoch
'''
plt.figure(5, figsize=(60, 40))
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


'''
Finally save the model.
'''
model.save('model.h5')


'''
Code to plot some graphs for representation purposes
'''
# Manually concatenated the Sample and generated training CSV file.
drivLog = pd.read_csv('./driving_log.csv', names=['Center', 'Left', 'Right', 'Steering Angle',
                                                  'Throttle', 'Brake', 'Speed'], header=0)

plt.figure(figsize=(20, 10))
drivLog.plot()

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
drivLog['Steering Angle'].plot(ax=axes[0], color='red')
axes[0].set_title('Steering Angle')
drivLog['Throttle'].plot(ax=axes[1], color='blue')
axes[1].set_title('Throttle')
drivLog['Brake'].plot(ax=axes[2], color='green')
axes[2].set_title('Brake')
drivLog['Speed'].plot(ax=axes[3], color='gray')
axes[3].set_title('Speed')
plt.show()


'''
Code to plot a few random images cropped the way Keras gets them.
'''
for i in range(5):
    img_nos = int(np.floor(random() * len(images)))
    print(img_nos)
    temp_img = cv2.imread(images[img_nos])
    cv2.imshow('OG_image', temp_img)
    crop_image = temp_img[50:140,:,:]
    cv2.imshow('Crop_image', crop_image)
    cv2.imshow('YUV_image', preprocess_image(crop_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

K.clear_session()
