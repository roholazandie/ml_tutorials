import tensorflow as tf
from tensorflow.python.ops.init_ops import RandomUniform
from tensorflow.python.client import device_lib
import numpy as np

def simple_properties1():
    #  Initialization from another variable
    weights = tf.Variable(tf.random_normal(shape=[10, 10], mean=5, stddev=0.1))

    # Create another variable with the same value as 'weights'
    w2 = tf.Variable(weights.initialized_value(), name="w2")

    # Create another variable with twice as many as the weights
    w3 = tf.Variable(weights.initialized_value() * 2, name="w3")

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        print(session.run(weights))
        print(session.run(w2))
        print(session.run(w3))


def get_variable_examples():
    my_variable = tf.get_variable("my_variable",
                                  shape=(2,3),
                                  initializer=tf.zeros_initializer)

    # Both of the approaches below correct and do the same thing
    initializer = RandomUniform(minval=-1, maxval=1)
    my_variable2 = tf.get_variable("my_variable2",
                                  shape=(2,3),
                                  initializer=initializer)

    my_variable3 = tf.get_variable("my_variable3",
                                  shape=(2,3),
                                  initializer=tf.random_uniform_initializer(-1, 1))

    my_variable4 = tf.get_variable("my_variable4",
                                   dtype=tf.int32,
                                   initializer=tf.constant(3, shape=(2,3)))
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        realized_variable1 = session.run(my_variable)
        realized_variable2 = session.run(my_variable2)
        realized_variable3 = session.run(my_variable3)
        realized_variable4 = session.run(my_variable4)
        print(realized_variable1)
        print(realized_variable2)
        print(realized_variable3)
        print(realized_variable4)


def variable_on_device():
    device = available_devices('CPU')
    with tf.device(device[0]):
        v = tf.get_variable("v1", [1])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(v))


def initialize_variables():
    # shape=() scalar 0.2
    # shape=(1) 1-d array(=tensor) [0.2]
    # shape=(1,1) 2-d array [[0.2]]
    v = tf.get_variable('v', shape=(), initializer=tf.random_normal_initializer(mean=3, stddev=0.1))
    w = tf.get_variable('w', initializer=v.initialized_value()+1)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        realized_v = session.run(v)
        realized_w = session.run(w)
        print(realized_v)
        print(realized_w)


def using_variables():
    '''
    All variables in tensorflow are symbolic variables
    they do not have any value unless we run a session to assign them
    In classic programming variables do hold values, and operations modify them
    In symbolic programming like tensorflow it's more about building a graph of operations
    that will be compiled later for execution.
    Such an architecture enables the code to be compiled and executed indifferently
    on CPU or GPU for example. Symbols are an abstraction that
    does not require to know where it will be executed.

    The second aspect of symbolic computing is that it is
     much more like mathematical functions or formulas
    '''
    v = tf.get_variable('v', shape=(), initializer=tf.zeros_initializer)
    #assignment = v.assign_add(1)
    print(v)
    v = v+3
    print(v)
    b = v.assign_add(4)
    w = v+1
    n = v*3
    # even this constant variable is a symbolic variable
    angle = tf.get_variable('angle', initializer=tf.constant(np.pi/2))
    s = tf.sin(angle)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(v))
     #   print(session.run(assignment))
        print(session.run(w))
        #print(session.run(b))
        print(session.run(n))
        print(session.run(s))


def variable_and_tensor_difference():
    '''
    Variable is basically a wrapper on Tensor that maintains state across multiple calls to run
    and a variable's value can be updated by backpropagation
    These differences mean that you should think of a variable as
    representing your model's trainable parameters (for example, the weights and biases of a neural network),
    while you can think of a Tensor as representing the data being
    fed into your model and the intermediate representations of that data as it passes through your model.
    '''
    tensor_variable = tf.zeros((3,4))
    with tf.Session() as session:
        print(session.run(tensor_variable))

    # Declaring variables can be done with tf.Variable or tf.get_variable
    variable = tf.Variable([3,5,6])
    # OR
    #variable = tf.get_variable('variable', shape=(3,), dtype=tf.int32)
    #variable.assign([3,5,6])


    w = variable+1 # w is a tf.Tensor which is computed based on value of variable
    with tf.Session() as session:
        # For variables we need to initialize the computation graph
        session.run(tf.global_variables_initializer())
        print(session.run(variable))
        print(session.run(w))

    # if you want the variable stays variable you should use use tensorflow ops on them
    variable2 = tf.Variable([1,3,2])
    variable3 = tf.add(variable, variable2)
    print(variable)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(variable3))

    # add constant
    variable4 = tf.add(variable, tf.constant(3))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(variable4))

    # sin, cos
    v = tf.Variable([4, 5, 2], dtype=tf.float32)
    # variable5 = 5*Sin(Cos(v))
    variable5 = tf.multiply(tf.sin(tf.cos(v)), tf.constant(5, dtype=tf.float32))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(variable5))


def difference_between_placeholder_and_variables():
    '''
    A placeholder is used for feeding external data into a Tensorflow computation
    If you're training a learning algorithm, the clear use case of placeholder is to feed in your
    training data. The training data isn't stored in the computation graph.
    How are you going to get it into the graph? By injecting it through a placeholder.
    A placeholder is basically you telling the graph "I don't have this for you yet.
    But I'll have it for you when I ask you to run."
    '''
    x = tf.placeholder(dtype=tf.float32, shape=(3,), name='x')
    session = tf.Session()
    print(session.run(x, feed_dict={x:np.array([2,5,3], dtype=np.float32)}))

    a = tf.Variable(initial_value=np.array([[1, 2], [4, 2]]), dtype=tf.float32, name='a')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(a))


def available_devices(device_name='CPU'):
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == device_name]


if __name__ == "__main__":
    #get_variable_examples()
    #variable_on_device()
    #initialize_variables()
    #using_variables()
    #variable_and_tensor_difference()
    difference_between_placeholder_and_variables()

