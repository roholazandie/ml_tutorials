import theano
import theano.tensor as th
import numpy as np
import tensorflow as tf

log_dir = "/home/rohola/tmp/tutorial_log_dir"

def simple_addition():
    '''
    this function shows the difference and similarities between
    theano and tensorflow in a simple addition
    the result of z = a+b are tensor in both of them.
    '''
    # Theano
    a = th.scalar(dtype='int32')
    b = th.scalar(dtype='int32')
    z = a+b
    print(type(z)) # theano.tensor.var.TensorVariable
    f = theano.function([a, b], z)
    print(f(10, 32))
    # Tensorflow
    a = tf.placeholder(tf.int32)
    b = tf.placeholder(tf.int32)
    z = a+b
    print(type(z)) # tensorflow.python.framework.ops.Tensor
    session = tf.Session()
    result = session.run(z, feed_dict={a:10, b:32})
    print(result)


def differentiation():
    a = th.scalar(dtype='float32')
    ga = th.grad(a**2, a)
    f = theano.function([a], ga)
    print(f(2))


def tensors_evaluation():
    # theano
    a = th.scalar(dtype='int32')
    b = th.scalar(dtype='int32')
    z = a + b
    print(z.eval({a: 10, b: 32}))

    # tensorflow
    a = tf.placeholder(dtype=tf.int32)
    b = tf.placeholder(dtype=tf.int32)
    z = a + b
    with tf.Session():
        print(z.eval({a: 10, b: 32}))


def tensors_with_different_types_and_shapes():
    # theano
    a = theano.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'), name='a')
    print(a.eval())

    # tensorflow
    a = tf.Variable(initial_value=np.array([[1, 2, 3],[4, 5, 6]]), dtype=tf.float32, name='a')
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(a))


def adding_two_matrix():
    # theano
    a = th.matrix(name='a', dtype='float32') # matrices are 2d tensors in theano
    b = th.matrix(name='b', dtype='float32')
    print(type(a)) # TensorVariable
    z = a + b
    f = theano.function([a, b], z)
    value = f(np.array([[1, 2], [4, 2]], dtype='float32'), np.array([[2, 7], [1, 9]], dtype='float32'))
    print(value)

    # theano
    a = theano.shared(value=np.array([[1, 2], [4, 2]], dtype='float32'), name='a')
    b = theano.shared(value=np.array([[2, 7], [1, 9]], dtype='float32'), name='b')
    z = a + b
    print(type(a)) # TensorSharedVariable
    print(a.get_value())
    print(z.eval())

    # tensorflow Variable is lower level, always create a variable
    a = tf.Variable(initial_value=np.array([[1, 2], [4, 2]]), dtype=tf.float32, name='a')
    b = tf.Variable(initial_value=np.array([[2, 7], [1, 9]]), dtype=tf.float32, name='b')
    z = tf.add(a, b)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(z))

    # tensorflow this is the recommenended way to create variables, easy to refactor
    # if the variable exist doesn't create it
    a = tf.get_variable(name='a', dtype=tf.float32, initializer=np.array([[1, 2], [4, 2]], dtype=np.float32))
    b = tf.get_variable(name='b', dtype=tf.float32, initializer=np.array([[2, 7], [1, 9]], dtype=np.float32))
    z = tf.add(a, b)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(z))


def visualize_computational_graphs():
    a = tf.get_variable(name='a', shape=(3,3), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=3, maxval=4))
    b = tf.get_variable(name='b', shape=(3,3), dtype=tf.float32, initializer=tf.random_normal_initializer(mean=3, stddev=0.1))
    c = tf.get_variable(name='c', shape=(3,3), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    z = tf.sin(a)+tf.cos(b*c+a)
    print(a.op)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(z)
        # visualization
        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        summary_writer.flush()




if __name__ == "__main__":
    #simple_addition()
    #differentiation()
    #tensors_evaluation()
    #tensors_with_different_types_and_shapes()
    #adding_two_matrix()
    visualize_computational_graphs()