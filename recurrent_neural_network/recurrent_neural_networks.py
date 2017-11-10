import numpy as np
import plotlyvisualization as visualization
import tensorflow as tf

num_steps = 5  # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


"""
RNN Inputs
"""
# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
def get_rnn_inputs(x):
    x_one_hot = tf.one_hot(x, num_classes)
    rnn_inputs = tf.unstack(x_one_hot, axis=1)
    return rnn_inputs


'''
Definition of RNN cell
'''
def rnn_cell(rnn_input, state, init_state):
    if state == init_state:
        with tf.variable_scope('rnn_cell'):
            W = tf.get_variable('W', [num_classes + state_size, state_size])
            b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    else:
        with tf.variable_scope('rnn_cell', reuse=True):
            W = tf.get_variable('W', [num_classes + state_size, state_size])
            b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


def get_rnn_outputs(rnn_inputs, init_state):
    state = init_state
    rnn_outputs = []
    for rnn_input in rnn_inputs:
        state = rnn_cell(rnn_input, state, init_state)
        rnn_outputs.append(state)
    final_state = rnn_outputs[-1]
    return rnn_outputs, final_state


# logits and predictions
def get_logits(rnn_outputs):
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    return logits


def get_predictions(logits):
    '''
    Just for generation
    :param logits:
    :return:
    '''
    predictions = [tf.nn.softmax(logit) for logit in logits]
    return predictions


def get_labels(y):
    # Turn our y placeholder into a list of labels
    y_as_list = tf.unstack(y, num=num_steps, axis=1)
    return y_as_list


# losses and train_step
def get_losses(logits, y_as_list):
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
              logit, label in zip(logits, y_as_list)]
    total_loss = tf.reduce_mean(losses)
    return losses, total_loss


def get_train_step(total_loss):
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
    return train_step


def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
    init_state = tf.zeros([batch_size, state_size])

    rnn_inputs = get_rnn_inputs(x)
    rnn_outputs, final_state = get_rnn_outputs(rnn_inputs, init_state)
    logits = get_logits(rnn_outputs)
    predictions = get_predictions(logits)
    losses, total_loss = get_losses(logits, y_as_list=get_labels(y))
    train_step = get_train_step(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            print("epoch: ", idx)
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = sess.run([losses,
                                                                         total_loss,
                                                                         final_state,
                                                                         train_step],
                                                                        feed_dict={x: X, y: Y,
                                                                                   init_state: training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss / 100)
                    training_losses.append(training_loss / 100)
                    training_loss = 0

    return training_losses


if __name__ == "__main__":
    training_losses = train_network(1, num_steps)
    visualization.plot(training_losses)

