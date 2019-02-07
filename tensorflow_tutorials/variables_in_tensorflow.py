import tensorflow as tf
from tensorflow.python.ops.init_ops import RandomUniform
from tensorflow.python.client import device_lib
import numpy as np

log_dir = "/home/rohola/tmp/tutorial_log_dir"


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
                                  shape=(2, 3),
                                  initializer=tf.zeros_initializer)

    # Both of the approaches below correct and do the same thing
    initializer = RandomUniform(minval=-1, maxval=1)
    my_variable2 = tf.get_variable("my_variable2",
                                   shape=(2, 3),
                                   initializer=initializer)

    my_variable3 = tf.get_variable("my_variable3",
                                   shape=(2, 3),
                                   initializer=tf.random_uniform_initializer(-1, 1))

    my_variable4 = tf.get_variable("my_variable4",
                                   dtype=tf.int32,
                                   initializer=tf.constant(3, shape=(2, 3)))
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
    w = tf.get_variable('w', initializer=v.initialized_value() + 1)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        realized_v = session.run(v)
        realized_w = session.run(w)
        print(realized_v)
        print(realized_w)


def using_variables():
    '''
    All variables in tensorflow are symbolic variables
    they do not have any value unless we run a session to variables_are_stateful them
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
    # assignment = v.assign_add(1)
    print(v)
    v = v + 3
    print(v)
    b = v.assign_add(4)
    w = v + 1
    n = v * 3
    # even this constant variable is a symbolic variable
    angle = tf.get_variable('angle', initializer=tf.constant(np.pi / 2))
    s = tf.sin(angle)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(v))
        #   print(session.run(assignment))
        print(session.run(w))
        # print(session.run(b))
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
    tensor_variable = tf.zeros((3, 4))
    with tf.Session() as session:
        print(session.run(tensor_variable))

    # Declaring variables can be done with tf.Variable or tf.get_variable
    variable = tf.Variable([3, 5, 6])
    # OR
    # variable = tf.get_variable('variable', shape=(3,), dtype=tf.int32)
    # variable.variables_are_stateful([3,5,6])


    w = variable + 1  # w is a tf.Tensor which is computed based on value of variable
    with tf.Session() as session:
        # For variables we need to initialize the computation graph
        session.run(tf.global_variables_initializer())
        print(session.run(variable))
        print(session.run(w))

    # if you want the variable stays variable you should use use tensorflow ops on them
    variable2 = tf.Variable([1, 3, 2])
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

    There is no such distinction in theano. In theano we have tensors for inputs and variables
    '''
    x = tf.placeholder(dtype=tf.float32,
                       name='x')  # no need to specify the size beforehand because we maybe don't know how many training instances we have
    session = tf.Session()
    print(session.run(x, feed_dict={x: np.array([2, 5, 3], dtype=np.float32)}))

    '''
    A variable is used to store state in your graph. 
    It requires an initial value. One use case could be representing weights of a neural network or something similar. 
    '''
    a = tf.Variable(initial_value=np.array([[1, 2], [4, 2]]), dtype=tf.float32,
                    name='a')  # should have size and then variables_are_stateful value or initial value(which obviously have size)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(a))


def graph_in_tensorflow():
    c = tf.Variable(4.0, name='c')
    g = tf.get_default_graph()
    operations = g.get_operations()
    print(operations[0].node_def)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # Visualize the graph
    summary_writer = tf.summary.FileWriter(log_dir, session.graph)
    summary_writer.flush()


def variables_are_stateful():
    a = tf.Variable(2, name='a')
    # Be careful we cannot write:
    # a = a*2
    # because the result of a*2 operation is no longer a Variable, it will be a immutable Tensor
    a_times_two = a.assign(a * 2)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        # Everytime we call the session.run() the variables
        # get updated the operation doesn't wipe data in them
        # Variable is a trick to make mutable data in tensorflow
        print(session.run(a_times_two))
        print(session.run(a_times_two))
        print(session.run(a_times_two))
        print(session.run(a_times_two))

        # Visualize the graph
        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        summary_writer.flush()


def tensor_transformation():
    def to_double_tensor():
        a = tf.constant(9, dtype=tf.int32)
        print(a)
        m = tf.to_double(a)
        print(m)

    def cast_tensor():
        a = tf.constant(9, dtype=tf.int32)
        m = tf.cast(a, dtype=tf.float32)
        print(m)

    def reshape_tensor():
        a = tf.constant(4, shape=(5, 6))
        print(a)
        m = tf.reshape(a, shape=(6, 5))
        print(m)

    def flatten_tensor():
        a = tf.random_uniform((3, 4))
        m = tf.reshape(a, shape=(12,))
        ######OR######
        n = tf.reshape(a, (-1,))
        print(m)
        print(n)

    def split_tensor():
        split0, split1, split2 = tf.split(tf.ones(shape=(5, 30)), [4, 15, 11], 1)
        print(tf.shape(split0))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(split0))
            print(sess.run(split1))

    # def tile_tensor():
    #     c = tf.constant(3, shape=(2, 3))
    #     t = tf.tile(c, [1,1,1])
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         print(sess.run(t))

    def concat_tensor():
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        t3 = tf.concat([t1, t2], 0)
        t4 = tf.concat([t1, t2], 1)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print(session.run(t3))
            print(session.run(t4))

    def stack_tensors():
        t1 = tf.constant(1, shape=(4,))
        t2 = tf.constant(2, shape=(4,))
        t3 = tf.stack([t1, t2])
        t4 = tf.stack([t1, t2], axis=1)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print(session.run(t3))
            print(session.run(t4))

    def strided_slice_tensor():
        #This is a good way to slice the tensor
        # we just need to specify the begin and end positions
        #First example
        l = tf.Variable([[1,2,3,4],[5,6,7,8]])
        c = tf.strided_slice(l, [0, 0], [1, 2]) #[1,2]
        b = tf.strided_slice(l, [0, 1], [2, 3]) #[[2 3], [6 7]]

        j = tf.Variable([[[12, 1, 1], [2, 2, 2]],
                         [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 6, 6]]])
        t = tf.strided_slice(j, [0, 0, 0], [3, 2, 2])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(b))
            print(sess.run(c))
            print(sess.run(t))


    def dynamic_partition_tensor():
        data = [5, 1, 7, 2, 3, 4]
        partitions = [0, 0, 1, 1, 0, 1]
        num_partitions = 2
        parts = tf.dynamic_partition(data=data, partitions=partitions, num_partitions=num_partitions)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print(session.run(parts))


    def one_hot_encoder():
        data = [[1,2,3], [2,1,3]]
        output = tf.one_hot(data, 5)
        output = tf.unstack(output)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(output))


    def unstak_tensor():
        x = [[1,2,3], [4,5,6]]
        output = tf.unstack(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(output))


    def multiplication_tensors():
        x = tf.ones([2, 3, 5])
        x1 = tf.reshape(x, [-1, 5]) # we don't care about the 2 and 3 dimensions(first and second) so those dimenstion will be flattened
        y = tf.ones((5, 4))
        c = tf.matmul(x1, y)
        c = tf.reshape(c, [2, 3, 4])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(c))
            print(sess.run(x1))

    # to_double_tensor()
    # cast_tensor()
    # reshape_tensor()
    #flatten_tensor()
    #split_tensor()
    # # tile_tensor()
    # concat_tensor()
    #stack_tensors()
    #strided_slice_tensor()
    #dynamic_partition_tensor()
    # one_hot_encoder()
    #unstak_tensor()
    multiplication_tensors()



def tensor_in_tensorflow():
    '''
    1- In the Python API, a tf.Tensor object represents the symbolic result of a TensorFlow operation.
    For example, in the expression t = tf.matmul(x, y), t is a tf.Tensor object
    representing the result of multiplying x and y
    (which may themselves be symbolic results of other operations, concrete values such as NumPy arrays, or variables).
    In this context, a "symbolic result" is more complicated than a pointer to the result of an operation.
    It is more analogous to a function object that, when called (i.e. passed to tf.Session.run())
    will run the necessary computation to produce the result of that operation, and return it to you as a concrete value (e.g. a NumPy array).

    2- In the C++ API, a tensorflow::Tensor object represents the concrete value of a multi-dimensional array.
    For example, the MatMul kernel takes two two-dimensional tensorflow::Tensor objects as inputs, and produces a single two-dimensional tensorflow::Tensor object as its output.

    This distinction is a little confusing, and we might choose different names if we started over
    (in other language APIs, we prefer the name Output for a symbolic result and Tensor for a concrete value).

    A similar distinction exists for variables. In the Python API, a tf.Variable is the symbolic representation of a variable,
    which has methods for creating operations that read the current value of the variable, and variables_are_stateful values to it.
    In the C++ implementation, a tensorflow::Var object is a wrapper around a shared, mutable tensorflow::Tensor object.


    -------------------------------------------------------
    Basically, every data is a Tensor in TensorFlow (hence the name):
    1- placeholders are Tensors to which you can feed a value (with the feed_dict argument in sess.run())
    2- Variables are Tensors which you can update (with var.variables_are_stateful()). Technically speaking, tf.Variable is not a subclass of tf.Tensor though
    3- tf.constant is just the most basic Tensor, which contains a fixed value given when you create it
    '''


def available_devices(device_name='CPU'):
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos ]


if __name__ == "__main__":
    print(available_devices())
    # get_variable_examples()
    # variable_on_device()
    # initialize_variables()
    # using_variables()
    # variable_and_tensor_difference()
    # difference_between_placeholder_and_variables()
    # graph_in_tensorflow()
    # variables_are_stateful()
    tensor_transformation()
