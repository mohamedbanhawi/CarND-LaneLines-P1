import cv2
import csv
import numpy as np
import json
import DataAugmentation
import matplotlib.pyplot as plt

DA = DataAugmentation.DataAugmentation()

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
            image = cv2.imread(image_path)
            images.append(image)
            steering_measurement = float(line[3])
            steering_measurements.append(steering_measurement)
            if data_augmentation['flip']:
                # flip
                image_flip = cv2.flip(image, 1)
                images.append(image_flip)
                steering_measurement_flip = -steering_measurement
                steering_measurements.append(steering_measurement_flip)

            if data_augmentation['use_left_camera']:
                # other cameras
                # create adjusted steering measurements for the side camera images
                correction = data_augmentation['camera_correction'] # this is a parameter to tune
                steering_left = steering_measurement + correction

                image_path_left = line[1]
                image_path_left = image_path_left.split('/')[-1]
                image_path_left = dataset + 'IMG/' +image_path_left
                image_left = cv2.imread(image_path_left)
                # add images and angles to data set
                images.append(image_left)
                steering_measurements.append(steering_left)

            if data_augmentation['use_right_camera']:
                # other cameras
                # create adjusted steering measurements for the side camera images
                correction = data_augmentation['camera_correction'] # this is a parameter to tune
                steering_right = steering_measurement - correction

                image_path_right = line[2]
                image_path_right = image_path_right.split('/')[-1]
                image_path_right = dataset + 'IMG/' +image_path_right
                image_right = cv2.imread(image_path_right)
                # add images and angles to data set
                images.append(image_right)
                steering_measurements.append(steering_right)

            if data_augmentation['blur']:
                kernel = data_augmentation['blur_kernel']
                image_blurred = DA.blur_dataset(image, kernel)
                images.append(image)
                steering_measurements.append(steering_measurement)

            if data_augmentation['brighten']:
                gamme = 5
                image_bright = DA.adjust_gamma(image, gamme)
                images.append(image_bright)
                steering_measurements.append(steering_measurement)

            if data_augmentation['darken']:
                gamme = 0.35
                image_dark = DA.adjust_gamma(image, gamme)
                images.append(image_dark)
                steering_measurements.append(steering_measurement)

            if data_augmentation['translate']:
                distance = data_augmentation['distance_px']
                image_pos = DA.shift_dataset(image, distance)
                images.append(image_pos)
                steering_measurements.append(steering_measurement)

                distance = -distance
                image_neg = DA.shift_dataset(image, distance)
                images.append(image_neg)
                steering_measurements.append(steering_measurement)

            if data_augmentation['rotate']:
                angle = data_augmentation['rotation_angle_deg']
                image_rotate_pos = DA.rotate_dataset(image, angle)
                images.append(image_rotate_pos)
                steering_measurements.append(steering_measurement)

                angle = -angle
                image_rotate_neg= DA.rotate_dataset(image, angle)
                images.append(image_rotate_neg)
                steering_measurements.append(steering_measurement)
            if data_augmentation['randomise']:
                n = data_augmentation['randomise_factor']
                for i in range(n):
                    image_rand = DA.random_jitter(image)
                    images.append(image_rand)
                    steering_measurements.append(steering_measurement)

if data_augmentation['show_images']: 
    cv2.imshow('Original Image',image)
    if data_augmentation['flip']:
        cv2.imshow('Flipped Image',image_flip)
    if data_augmentation['use_left_camera']:
        cv2.imshow('Left Camera',image_left)
    if data_augmentation['use_right_camera']:
        cv2.imshow('Right Camera',image_right)
    if data_augmentation['blur']:
        cv2.imshow('Blurred',image_blurred)
    if data_augmentation['brighten']:
        cv2.imshow('Bright',image_bright)
    if data_augmentation['darken']:
        cv2.imshow('Dark',image_dark)
    if data_augmentation['translate']:
        cv2.imshow('Translate Positive',image_pos)
        cv2.imshow('Translate Negative',image_neg)
    if data_augmentation['rotate']:
        cv2.imshow('Rotate Pos',image_rotate_pos)
        cv2.imshow('Rotate - Neg ',image_rotate_neg)
    if data_augmentation['randomise']:
        cv2.imshow('Randomise', image_rand)

    key = cv2.waitKey(0)
    if key == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

if data_augmentation['write_images']: 
    cv2.imwrite('Original Image.png',image)
    if data_augmentation['flip']:
        cv2.imwrite('Flipped Image.png',image_flip)
    if data_augmentation['use_left_camera']:
        cv2.imwrite('Left Camera.png',image_left)
    if data_augmentation['use_right_camera']:
        cv2.imwrite('Right Camera.png',image_right)
    if data_augmentation['blur']:
        cv2.imwrite('Blurred.png',image_blurred)
    if data_augmentation['brighten']:
        cv2.imwrite('Bright.png',image_bright)
    if data_augmentation['darken']:
        cv2.imwrite('Dark.png',image_dark)
    if data_augmentation['translate']:
        cv2.imwrite('Translate Positive.png',image_pos)
        cv2.imwrite('Translate Negative.png',image_neg)
    if data_augmentation['rotate']:
        cv2.imwrite('Rotate Pos.png',image_rotate_pos)
        cv2.imwrite('Rotate - Neg.png',image_rotate_neg)
    if data_augmentation['randomise']:
        cv2.imwrite('Randomise.png', image_rand)


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

if training_parameters['network'] == 'nvidia':
    # implement the end to end driving network by nvidia
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

else if training_parameters['network'] == lenet:
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')

history_object = model.fit(X_train, y_train, nb_epochs=training_parameters['epochs'], validation_split = 0.2, shuffle = True)

model.save(training_parameters['network']+'_cloning_model.h5')

if training_parameters['visualise_performance']:
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()






