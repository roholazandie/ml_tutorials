import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#training data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
learning_rate = 0.01

# Placeholders
X = tf.placeholder(dtype=tf.float32, name='X')
Y = tf.placeholder(dtype=tf.float32, name='Y')

# Varilables
W = tf.get_variable(shape=(1,), name="W")
b = tf.get_variable(shape=(1,), name="b")


# Defining the model
prediction = tf.multiply(W, X) + b

losses = tf.pow(prediction-Y, 2)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)


W_value = 0
b_value = 0
num_epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        for x, y in zip(train_X, train_Y):

            iteration_loss, _, W_value, b_value = sess.run([total_loss, train_step, W, b], feed_dict={X: x, Y:y})
            #print(iteration_loss)
            print(W_value, b_value)



plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, W_value * train_X + b_value, label='Fitted line')
plt.legend()
plt.show()