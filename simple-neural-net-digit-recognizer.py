# This tensorflow program classifies handwritten digits from the MNIST dataset using a simple Neural Network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

### STEP 1: Read in training data set ###

# Tensorflow has a built-in helper function to pull down the dataset, since it's a classic training set.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


### STEP 2: Set up TF containers ###

# Define a tensor placeholder for the image. Each image is greyscale at 28 x 28 pixels.
# The first parameter to shape is None because the batch size can be any size.
# The second parameter to shape is the size of one image. It is represented as a 784 (28*28) item vector.
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ ("y bar") is the probability vector containing the probabilities that the image is the given digit (0-9)
y_ = tf.placeholder(tf.float32, [None, 10])  

# We'll have 10 neurons in this neural net: one for each possible digit. 
# There will be 784 inputs to each neuron.
# We're going to train the weights (one for each input for each neuron - 784 x 10)
W = tf.Variable(tf.zeros([784, 10]))
# And we're going to train the biases (one for each neuron) 
b = tf.Variable(tf.zeros([10]))


### STEP 3: Set up learning model ###

# We'll use softmax as our activation function since it's great for categorical distributions.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross entropy is a log loss function that penalizes confident wrong answers strongly.
# This is also handy for categorical distributions.
cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# The training will minimize cross entropy. We'll do this with Gradient Descent.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


### STEP 4: Launch training session ###

# Initialize TF global variables.
init = tf.global_variables_initializer()

# Start the session.
sess = tf.Session()
sess.run(init)

# We'll do 1000 iterations with a batch size of 100 images.
for i in range(1000):
    # batch_xs is the image vectors, batch_ys is the classification.
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Run the optimizer.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Compare the actual highest probability classification to our model.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()