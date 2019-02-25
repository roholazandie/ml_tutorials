import tensorflow as tf
import numpy as np

width = 8
height = 8
batch_size = 100
filter_height = 3
filter_width = 3
in_channels = 3
channel_multiplier = 1
out_channels = 3


input_tensor = tf.get_variable(shape=(batch_size, height, width, in_channels), name="input")
depthwise_filter = tf.get_variable(shape=(filter_height, filter_width, in_channels, channel_multiplier), name="deptwise_filter")
pointwise_filter = tf.get_variable(shape=[1, 1, channel_multiplier * in_channels, out_channels], name="pointwise_filter")

output = tf.nn.separable_conv2d(
    input_tensor,
    depthwise_filter,
    pointwise_filter,
    strides=[1,1,1,1],
    padding='SAME',
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    output_value = sess.run(output, feed_dict={input_tensor: np.random.rand(batch_size, width, height, in_channels),
                                               depthwise_filter: np.random.rand(filter_height, filter_width, in_channels, channel_multiplier),
                                               pointwise_filter: np.random.rand(1, 1, channel_multiplier * in_channels, out_channels)})
    print(np.shape(output_value))




def conv1d_and_dense_layer_comparison():
    from tensorflow.python.keras.layers import SeparableConv1D

    SeparableConv1D()