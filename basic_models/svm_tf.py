from sklearn import cross_validation
from sklearn import datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import plotlyvisualization as plot

def extract_data(filename):

    out = np.loadtxt(filename, delimiter=',')

    # Arrays to hold the labels and feature vectors.
    labels = out[:,0]
    labels = labels.reshape(labels.size,1)
    fvecs = out[:,1:]

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs,labels

X_train, y_train = extract_data("linearly_separable_data.csv")
y_train[y_train == 0] = -1
y_train = np.reshape(y_train, (1, len(y_train))).flatten()

# iris = datasets.load_iris()
# X_train = iris.data[:, 1:3]
# X_train = preprocessing.scale(X_train)
# y_train = iris.target
# y_train = np.array([1 if item==0 else -1 for item in y_train])


d = np.shape(X_train)[1] # dimension of data points
N = np.shape(X_train)[0] # number of data points
print(d)

X = tf.placeholder(dtype=tf.float32, name="X", shape=(d,1))
y = tf.placeholder(dtype=tf.float32, name="y", shape=(1,))
C = tf.placeholder(dtype=tf.float32, name="C", shape=(1,))

w = tf.get_variable(name="w", shape=(1, d))
b = tf.get_variable(name="b", shape=(1,))

#
m = 1-y *( tf.matmul(w, X)+b)
loss =  0.5*tf.pow(tf.norm(w), 2) + tf.maximum(0.0, m)


train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

num_epochs = 10


losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for _ in range(num_epochs):
    #     loss_value, _ = sess.run([loss, train_step], feed_dict={X: X_train, y: y_train})
    #     print(loss_value)
    for i in range(num_epochs):
        for xi, yi in zip(X_train, y_train):
            xi = np.reshape(xi, (2, 1))
            loss_value, _, w_value, b_value, m_value = sess.run([loss , train_step, w, b, m], feed_dict={X: xi, y: [yi]})
            #print(m_value)
            #print(w_value[0][0]/w_value[0][1])
            print(b_value)
            #print(i)
            #print(loss_value)
            #losses.append(loss_value[0][0])
            #print(loss_value[0][0])



plt.plot(X_train[np.where(y_train==1), 0], X_train[np.where(y_train==1),1], 'bo')
plt.plot(X_train[np.where(y_train==-1), 0], X_train[np.where(y_train==-1),1], 'ro')
x_line = np.linspace(min(X_train[:,0]), max(X_train[:,0]), 1000)
w_value = w_value[0]
print(w_value)


plt.plot(x_line, -(w_value[1]/w_value[0]) * x_line - b_value[0], label='Fitted line')
plt.legend()
plt.show()