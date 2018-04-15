import numpy as np
import matplotlib.pyplot as plt

# use MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)


# learning dataset
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

"""
This determines the size of the hidden layer in the neural network.
This is also known as the width of a layer.
"""
n_hidden_layer = 256 # number of hidden layers

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""
The MNIST data is made up of 28px by 28px images with a single channel. 
The tf.reshape() function above reshapes the 28px by 28px matrices in x 
into row vectors of 784px.
"""

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

# Combining linear functions together using a ReLU will give you a two layer network.
# xw + b
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']),\
    biases['hidden_layer'])
# Hidden layer with RELU activation
layer_1 = tf.nn.relu(layer_1)

# probability to keep units
keep_prob = tf.placeholder(tf.float32) 

# dropout neurons radnomly
layer_1 = tf.nn.dropout(layer_1, keep_prob)

# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])


# Define loss and optimizer
# cross entropy
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
# minimize cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
count = 0
# loss
loss = np.array([])
acc  = np.array([])
count_list = np.array([])
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            count += 1
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost_f, valid_accuracy = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, 
                                                                                         y: batch_y, 
                                                                                         keep_prob=0.5})
            if epoch % 50 == 0:
                # Print loss
                print('Loop: {} - Loss: {} - Validation Accuracy: {}'.format(count, cost_f, valid_accuracy))
            # store for plotting
            loss = np.append(loss, cost_f)
            acc = np.append(acc, valid_accuracy)
            count_list = np.append(count_list, count)

plt.subplot(1,2,1)
plt.plot(count_list, loss, 'b')
plt.ylabel('Cost')
plt.subplot(1,2,2)
plt.plot(count_list, acc , 'r')
plt.ylabel('Validation Accuracy')
plt.show()
