from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
log_dir = "/tmp/tensorflow/mnist/logs/sample"

# Actual deep learning
def weight_variable(shape):
  with tf.name_scope('hidden1'):
    initial = tf.truncated_normal(shape, stddev=0.1, name="weight1")
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def run_deep_nn_with_interactive_session():
    # Image palceholder and preperation
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch, width, height, channels]
    #  batch=-1 means the batch size will be adapted with different sizes that fed to the network
    y_ = tf.placeholder(tf.float32, [None, 10])

    # First Convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])  # [filter_height, filter_width, in_channels, out_channels]
    b_conv1 = bias_variable([32])  # the bias is for each output channels

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print("h_conv1")
    print(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    print("h_pool1")
    print(h_pool1)

    # Second Convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])  # the bias should always be the same as the output channel

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print("h_conv2")
    print(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print("h_pool2")
    print(h_pool2)

    # Densely Connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print("h_fc1")
    print(h_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print("y_conv")
    print(y_conv)

    # Train and evaluate the model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    for i in range(200):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                 y_: batch[1],
                                                 keep_prob: 1.0})

        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def run_deep_nn_with_session():
    # Image placeholder
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch, width, height, channels]
    y_ = tf.placeholder(tf.float32, [None, 10])

    # First Convolutional layer

    W_conv1 = weight_variable([5, 5, 1, 32])  # [filter_height, filter_width, in_channels, out_channels]
    b_conv1 = bias_variable([32])  # the bias is for each output channels

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    # Second Convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])  # the bias should always be the same as the output channel

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Train and evaluate the model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step_operation = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(200):
            batch = mnist.train.next_batch(50)
            feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            # train_step_operation.run(feed_dict=feed_dict)
            _, loss_value = session.run([train_step_operation, cross_entropy], feed_dict=feed_dict)
            print("loss_value %f" % (loss_value))

            if i % 100 == 0:
                feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
                train_accuracy = accuracy.eval(feed_dict=feed_dict)

                print("step %d, training accuracy %g" % (i, train_accuracy))

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
    #run_deep_nn_with_interactive_session()
    run_deep_nn_with_session()
