import cv2
import csv
import numpy as np

lines = []
images = []
steering_measurements = []

datasets = ['data/counterclockwise/']
log_name = 'driving_log.csv'

# load files
for dataset in datasets:
    with open(dataset+log_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            image_path = line[0]
            image_path = image_path.split('/')[-1]
            image_path = dataset + image_path
            image = cv2.imread(image_path)
            images.append(image)
            steering_measurement = float(line[3])
            steering_measurements.append(steering_measurement)
            # flip
            image = cv2.flip(image, 1)
            images.append(image)
            steering_measurement = -steering_measurement
            steering_measurements.append(steering_measurement)

X_train = np.array(images)
y_train = np.array(steering_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')

model.fit(X_train, y_train, validation_split = 0.2, shuffle = true)

model.save('cloning_model.h5')

