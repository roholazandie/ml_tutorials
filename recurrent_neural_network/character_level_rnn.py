import urllib.request
import os
import tensorflow as tf
import numpy as np
import reader
from tensorflow.contrib import rnn



log_dir = "/home/rohola/tmp/rnn_basic"


def read_data(file_name, file_url):
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(file_url, file_name)

    with open(file_name, 'r') as f:
        raw_data = f.read()
        print("Data length:", len(raw_data))

    return raw_data


def build_dataset(raw_data):
    raw_data = raw_data.strip('\n')
    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    data = [vocab_to_idx[c] for c in raw_data]
    return vocab_to_idx, idx_to_vocab, data, vocab_size



def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)


def build_basic_rnn_graph_without_list(
        num_classes,
        state_size=100,
        batch_size=32,
        num_steps=200,
        learning_rate=1e-4):
    # reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)

    rnn_inputs = x_one_hot

    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    #cell = tf.nn.rnn_cell.LSTMCell(state_size)
    # cell = tf.nn.rnn_cell.GRUCell(state_size)
    #cell = CustomCell(state_size, num_weights=5)
    #cell = LayerNormalizedLSTMCell(num_units=state_size)

    init_state = cell.zero_state(batch_size, tf.float32)

    output, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    #output, final_state = tf.nn.static_rnn(cell, rnn_inputs, dtype=tf.float32)
    output = tf.reshape(output, [-1, state_size])
    ##########OR########
    # outputs = []
    # state = init_state
    # with tf.variable_scope("RNN"):
    #     for time_step in range(num_steps):
    #         if time_step > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         (cell_output, state) = cell(rnn_inputs[:, time_step, :], state)
    #         outputs.append(cell_output)
    # final_state = state
    # output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, state_size])
    ###################

    output = tf.nn.dropout(output, keep_prob=0.5)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(output, W) + b
    logits = tf.reshape(logits, [batch_size, num_steps, num_classes])

    predictions = tf.nn.softmax(logits)

    loss_weights = tf.ones([batch_size, num_steps])
    losses = tf.contrib.seq2seq.sequence_loss(logits, y, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        final_state=final_state,
        init_state = init_state,
        total_loss=total_loss,
        train_step=train_step,
        logits = logits,
        predictions = predictions,
        saver=tf.train.Saver()
    )


def build_basic_rnn_graph_with_list(
        num_classes,
        state_size=100,
        batch_size=32,
        num_steps=200,
        learning_rate=1e-4):
    '''
    This function runs in legacy mode because instead of using tensors
    it uses the list of samples to calculate the loss function, this is so inefficient
    :param num_classes:
    :param state_size:
    :param batch_size:
    :param num_steps:
    :param learning_rate:
    :return:
    '''

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)
    rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, axis=1)]

    # Adding dropout
    #keep_prob = 0.7
    #rnn_inputs = [tf.nn.dropout(rnn_input, keep_prob) for rnn_input in rnn_inputs]

    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    #cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(state_size), rnn.BasicLSTMCell(state_size)])

    init_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = rnn.static_rnn(cell, rnn_inputs, dtype=tf.float32)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, state_size])

    # Adding dropout
    #output = tf.nn.dropout(output, keep_prob=0.5)
    ################
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes], initializer=tf.random_normal_initializer)
        b = tf.get_variable('b', [num_classes], initializer=tf.random_normal_initializer)

    logits = tf.matmul(output, W) + b

    predictions = tf.nn.softmax(logits)

    logits = tf.reshape(logits, [batch_size, num_steps, num_classes])
    logits = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(logits, num_steps, axis=1)]


    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, axis=1)]

    loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)

    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        final_state = final_state,
        init_state = init_state,
        total_loss = total_loss,
        train_step = train_step,
        predictions = predictions,
        logits = logits,
        saver = tf.train.Saver()
    )



def train_network(g, num_epochs, num_steps=200, batch_size=32, verbose=True, save=""):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        total_training_loss = 0
        steps = 0
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_state = None
            for X, Y in epoch:
                try:
                    steps += 1

                    feed_dict = {g['x']: X, g['y']: Y}

                    #if training_state is not None:
                    #    feed_dict[g['init_state']] = training_state

                    training_loss, _ , training_state, y, logits= sess.run([g['total_loss'], g['train_step'],g['init_state'], g['y'], g['logits']], feed_dict)
                    y = y[0]
                    logits = np.argmax(logits[0],1)
                    #logits = [np.argmax(item[0]) for item in logits]
                    print(logits, y)
                    total_training_loss += training_loss
                    #if verbose and idx % 5:
                    #    print("Average training loss for iteration", idx, " :", total_training_loss / steps)

                except:
                   print("exception occurred at "+str(idx))
                   continue
            #if save:
            #g['saver'].save(sess, save)

        #if save:
        #print("saved")
        g['saver'].save(sess, save)
        #########################################
        num_chars = 300
        input_chars = ['h', 'e', 'l', 'l', 'o', ' '] * 1000
        input_chars = input_chars[0:num_steps]

        current_chars = [vocab_to_idx[c] for c in input_chars]
        chars = current_chars

        state = None
        for i in range(num_chars):
            current_chars = np.reshape(current_chars, (1, num_steps))

            if state is not None:
                feed_dict = {g['x']: current_chars, g['init_state']: state}
            else:
                feed_dict = {g['x']: current_chars}

            predictions, state = sess.run([g['predictions'], g['final_state']], feed_dict)

            # current_chars = [np.random.choice(vocab_size, 1, p=np.squeeze(prediction))[0] for prediction in predictions]

            current_chars = [np.argmax(prediction, 1) for prediction in predictions]
            chars += list(current_chars[0])
            #current_chars = [np.argmax(prediction) for prediction in predictions]
            #chars += list(current_chars)

        chars = map(lambda x: idx_to_vocab[x], chars)
        print("".join(chars))

    return training_losses


if __name__ == "__main__":
    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file_name = 'data/hello1.txt'
    raw_data = read_data(file_name, file_url)
    vocab_to_idx, idx_to_vocab, data, vocab_size = build_dataset(raw_data)


    state_size = 200
    batch_size = 1
    num_steps = 100
    num_epochs = 1000

    g = build_basic_rnn_graph_without_list(vocab_size,
                                           state_size=state_size,
                                           batch_size=batch_size,
                                           num_steps=num_steps)

    # g = build_basic_rnn_graph_with_list(vocab_size,
    #                                     state_size=state_size,
    #                                     batch_size=batch_size,
    #                                     num_steps=num_steps)


    train_network(g, num_epochs=num_epochs, num_steps=num_steps, batch_size=batch_size, save=log_dir)
