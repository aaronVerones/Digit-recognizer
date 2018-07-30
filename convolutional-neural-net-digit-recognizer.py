# This tensorflow program classifies handwritten digits from the MNIST dataset using a Convolutional Neural Network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

### STEP 1: Read in training data set ###

# Tensorflow has a built-in helper function to pull down the dataset, since it's a classic training set.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Using Interactive session makes it the default sessions so we do not need to pass sess.
sess = tf.InteractiveSession()

### STEP 2: Set up TF containers ###

# Define a tensor placeholder for the image. Each image is greyscale at 28 x 28 pixels.
# The first parameter to shape is None because the batch size can be any size.
# The second parameter to shape is the size of one image. It is represented as a 784 (28*28) item vector.
x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ ("y bar") is the probability vector containing the probabilities that the image is the given digit (0-9)
y_ = tf.placeholder(tf.float32, [None, 10])


# This Convolutional Neural Network will try to figure out various features of the sample images.
# For this to work, we need to reshape the data into a tensor that represents the image more clearly than 
# a 784-item vector. This is a 28 pixel X 28 pixel X 1 grayscale value cube.
x_image = tf.reshape(x, [-1,28,28,1], name="x_image")

### STEP 3: Helper functions ###

# Define helper functions to created weights and baises variables, and convolution, and pooling layers
#   We are using RELU as our activation function.  These must be initialized to a small positive number 
#   and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling - we do Convolution, and then pooling to control overfitting
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

### STEP 4: Set up the layers of the CNN ###

# 1st Convolution layer
# Our first convolution layer goes through 5x5 chunks of the images, takes in 1 input (greyscale value)
# and produces 32 features.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# We're using RELU as our activation function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Reduce dimensionality through pooling. Also reduces sensitivity to outliers. Output is a 14x14 image.
h_pool1 = max_pool_2x2(h_conv1)

# 2nd Convolution layer
# Process the 32 features from Convolution layer 1, in 5 X 5 patch.  Return 64 features weights and biases.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# Do convolution of the output of the 1st convolution layer.  Pool results. Output is a 7x7 image.
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Using a Fully Connected Layer to connect our 7x7 images with 64 features to 1024 neurons.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Take 2nd Convolution Layer's output, flatten it, and pass it into Fully Connected Layer.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Randomly drop some of the Fully Connected Layer's neurons to prevent overfitting. 
# keep_prob is a placeholder so we can train this probability too. 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define model.
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Using cross entropy for loss measurement just like in the simple Neural Network implementation.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Adam Optimizer is a modification of Gradient Descent that varies the step size to prevent overshooting.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Correctness and Accuracy.
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all of the variables.
sess.run(tf.global_variables_initializer())

### STEP 5: Launch Training ###

# Number of steps, and how often to display the training progress.
num_steps = 3000
display_every = 50

# Start timer.
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Display training status every display_every steps.
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))


# Print summary of training.

# Elapsed time.
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

# Accuracy of our model.
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

sess.close()
