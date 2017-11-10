import theano
import theano.tensor as th
import numpy as np
import tensorflow as tf
from theano.gradient import jacobian


def simple_differentation():
    '''
    Both theano and tensorflow use symbolic differentation
    '''
    # Tensorflow
    x = tf.Variable(5.0)
    y = tf.square(x)  # y=x**2
    z = tf.gradients([y], [x])  # dy/dx = 2*x

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(z))

    # Theano
    x = th.scalar('x', dtype='float32')
    y = x**2
    z = th.grad(y, x)
    f = theano.function([x], z)
    print(f(5.0))


def jacobian1():
    x = th.vector('x', dtype='float32')
    y = x**2
    J, updates = theano.scan(lambda i, y, x: th.grad(y[i], x), sequences=th.arange(y.shape[0]), non_sequences=[y, x])
    f = theano.function([x], J, updates=updates)
    print(f([4, 5, 6]))

    # z = x**2+y**2
    # y_J = jacobian(z, [x,y])
    # f = theano.function([x,y], y_J)
    # print(f([4,5],[4,5]))


if __name__ == "__main__":
    #simple_differentation()
    jacobian1()