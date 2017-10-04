'''
This is tutorial from the book Hands-on Machine learning with scikit-learn and tensorflow (orielly)
'''
import tensorflow as tf
import numpy as np

def basic_rnn_in_tensorflow():
    # Data               intance0   instance1  instance2  instance3
    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

    # Model
    n_inputs = 3
    n_neurons = 5
    #n_timesteps = 2 # but this is implicit

    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])

    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons]), dtype=tf.float32)
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons]), dtype=tf.float32)
    b = tf.Variable(tf.zeros(shape=[1, n_neurons]), dtype=tf.float32)

    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Y0_value, Y1_value = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
        print(Y0_value)
        print(Y1_value)

        '''
            t = 0 
        [[-0.38679388 -0.3223801   0.49171168  0.99850345 -0.91205525] # instance 0(n_neuron long)
         [-0.97571319 -0.99962932  0.9999969   1.         -0.99998909] # instance 1
         [-0.99931675 -0.99999988  1.          1.         -1.        ] # instance 2
         [-1.         -1.          1.          0.99943352 -0.97268009]] # instance 3
        
            t = 1
        [[-0.99999624 -0.99999988  1.          1.         -1.        ]
         [-0.9972077   0.99985337 -0.8909409   0.96984023  0.34042129]
         [-0.99999523 -0.9930141   1.          1.         -0.99999887]
         [-0.99981439  0.80052388  0.99954522  0.99999607 -0.99120426]]
        '''

def static_unrolling_through_time():
    '''
    this function do exactly the same as basic_rnn_in_tensorflow() but with
    static_rnn to unroll through time
    :return:
    '''
    # Data               intance0   instance1  instance2  instance3
    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])  # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])  # t = 1

    # Model
    n_inputs = 3
    n_neurons = 5
    # n_timesteps = 2 # but this is implicit

    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])

    basic_cell = tf.contrib.rnn.BasicRNNCell(n_neurons)
    output_seq, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

    Y0, Y1 = output_seq
    # the final state(=states) for BasicRNNCell is equal to the last output(=Y1)

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Y0_value, Y1_value, states_value = sess.run([Y0, Y1, states], feed_dict={X0: X0_batch, X1: X1_batch})
        print(Y0_value)
        print(Y1_value)
        print(states_value)


def using_tensors_to_represent_timesteps():
    # Data                 t=0          t=1
    X_batch = np.array([[[0, 1, 2], [9, 8, 7]], # instance 0
                        [[3, 4, 5], [0, 0, 0]], # instance 1
                        [[6, 7, 8], [6, 5, 4]], # instance 2
                        [[9, 0, 1], [3, 2, 1]]]) # instance 3

    # Model
    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

    basic_cell = tf.contrib.rnn.BasicRNNCell(n_neurons)
    output_seq, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    outputs = tf.transpose(tf.stack(output_seq), perm=[1, 0, 2])

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_value, states_value = sess.run([outputs, states], feed_dict={X: X_batch})
        print(outputs_value)
        #print(states_value)


def dynamic_unrolling_through_time():
    # Data                 t=0          t=1
    X_batch = np.array([[[0, 1, 2], [9, 8, 7]],  # instance 0
                        [[3, 4, 5], [0, 0, 0]],  # instance 1
                        [[6, 7, 8], [6, 5, 4]],  # instance 2
                        [[9, 0, 1], [3, 2, 1]]])  # instance 3

    # Model
    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

    basic_cell = tf.contrib.rnn.BasicRNNCell(n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_value, states_value = sess.run([outputs, states], feed_dict={X: X_batch})
        print(outputs_value)


def handling_variable_lenth_input_sequences():
    '''
    the instance 1 data have the length 1 in comparison to others which have length 2
    actually we padded the input sequence to zero, but actually the seq_length ensures that
    it never fed the rnn
    :return:
    '''
    # Data                 t=0          t=1
    X_batch = np.array([[[0, 1, 2], [9, 8, 7]],  # instance 0
                        [[3, 4, 5], [0, 0, 0]],  # instance 1 padded with zero vector
                        [[6, 7, 8], [6, 5, 4]],  # instance 2
                        [[9, 0, 1], [3, 2, 1]]])  # instance 3

    seq_length_batch = np.array([2, 1, 2, 2])


    # Model
    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    seq_length = tf.placeholder(tf.int32, [None])

    basic_cell = tf.contrib.rnn.BasicRNNCell(n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, sequence_length=seq_length, dtype=tf.float32)

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_value, states_value = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
        #print(outputs_value)
        print(states_value)

    '''
    [[[-0.42685646 -0.89740378 -0.67770565  0.93859905 -0.33086076]
    [ 0.93208092 -1.          0.99996012  0.99999654 -0.99568522]] # final state

    [[-0.26243401 -0.99999279  0.13528825  0.9997977  -0.93367344] #final state
    [ 0.          0.          0.          0.          0.        ]]

    [[-0.08121563 -1.          0.79945654  0.99999934 -0.99533081]
    [ 0.36869711 -1.          0.9982937   0.99841589 -0.99564654]] # final state
    
    [[ 0.76992118 -0.99994683  0.99972117 -0.76455879 -0.99812895]
    [ 0.40788239 -0.99837512  0.78370684  0.50437582 -0.96873242]]] # final state
    '''





if __name__ == "__main__":
    #basic_rnn_in_tensorflow()
    #static_unrolling_through_time()
    #using_tensors_to_represent_timesteps()
    dynamic_unrolling_through_time()
    #handling_variable_lenth_input_sequences()