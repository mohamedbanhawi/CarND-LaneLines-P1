# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('small_traffic_set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# 32 CN, 3,3 Kernel, input is 32,32, 3 channels
# keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', 
# activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', 
# W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3))) 
# 2x2 Kernel
# default: keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))



# preprocess data (zero center)
X_normalized = np.array(X_train / 255.0 - 0.5 )

# One hot encoding
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

with open('small_traffic_set/small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

# Evaluate the test data in Keras Here
metrics = model.evaluate(X_normalized_test, y_one_hot_test, batch_size=32, verbose=1, sample_weight=None)
for metric_i in range(len(model.metrics_names)):
   metric_name = model.metrics_names[metric_i]
   metric_value = metrics[metric_i]
   print('{}: {}'.format(metric_name, metric_value))
