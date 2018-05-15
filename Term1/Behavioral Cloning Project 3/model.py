import cv2
import csv
import numpy as np
import json
import DataAugmentation
import matplotlib.pyplot as plt

with open('parameters.json') as f:
    parameters = json.load(f)

lines = []
images = []
steering_measurements = []

datasets = parameters['data_sets']
log_name = parameters['log_name']
data_augmentation = parameters['data_augmentation']
training_parameters = parameters['training']

# load files
for dataset in datasets:
    with open(dataset+log_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            image_path = line[0]
            image_path = image_path.split('/')[-1]
            image_path = dataset + 'IMG/' +image_path
            print (image_path)
            image = cv2.imread(image_path)
            images.append(image)
            steering_measurement = float(line[3])
            steering_measurements.append(steering_measurement)
            # flip
            image_flip = cv2.flip(image, 1)
            images.append(image_flip)
            steering_measurement_flip = -steering_measurement
            steering_measurements.append(steering_measurement_flip)

            if data_augmentation['use_left_camera']:
                # other cameras
                # create adjusted steering measurements for the side camera images
                correction = data_augmentation['camera_correction'] # this is a parameter to tune
                steering_left = steering_center + correction

                image_path_left = line[1]
                image_path_left = image_path_left.split('/')[-1]
                image_path_left = dataset + 'IMG/' +image_path_left

                # add images and angles to data set
                images.append(image_path_left)
                steering_measurements.append(steering_left)

            if data_augmentation['use_right_camera']:
                # other cameras
                # create adjusted steering measurements for the side camera images
                correction = data_augmentation['camera_correction'] # this is a parameter to tune
                steering_right = steering_center - correction

                image_path_right = line[2]
                image_path_right = image_path_left.split('/')[-1]
                image_path_right = dataset + 'IMG/' +image_path_right

                # add images and angles to data set
                images.append(image_path_right)
                steering_measurements.append(steering_right)

            if data_augmentation['blur']:
                image_blurred = DataAugmentation.blur_dataset(image)
                images.append(image)
                steering_measurements.append(steering_measurement)

            if data_augmentation['brighten']:
                gamme = 5
                image_bright = DataAugmentation.adjust_gamma(image, gamme)
                images.append(image_bright)
                steering_measurements.append(steering_measurement)

            if data_augmentation['darken']:
                gamme = 0.35
                image_dark = DataAugmentation.adjust_gamma(image_dark, gamme)
                images.append(image_dark)
                steering_measurements.append(steering_measurement)

            if data_augmentation['translate']:
                gamme = 0.35
                image_dark = DataAugmentation.adjust_gamma(image_dark, gamme)
                images.append(image_dark)
                steering_measurements.append(steering_measurement)

            if data_augmentation['rotate']:
                gamme = 0.35
                image_dark = DataAugmentation.adjust_gamma(image_dark, gamme)
                images.append(image_dark)
                steering_measurements.append(steering_measurement)

X_train = np.array(images)
y_train = np.array(steering_measurements)

shape = X_train[-1].shape

from keras.models import Sequential, Convolution2D, Model
from keras.layers import Flatten, Dense, Lambda, Activation

model = Sequential()
# crop
# remove bonnet and features from outside the road
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
# normalise
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape))
# implement the end to end driving network by nvidie
model.add(Convolution2D(24,5,5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3, activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3, activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('tanh'))

model.compile(loss='mse', optimizer = 'adam')

history_object = model.fit(X_train, y_train, nb_epochs=10, validation_split = 0.2, shuffle = True)

model.save('cloning_model.h5')

if training['visualise_performance']:
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()






