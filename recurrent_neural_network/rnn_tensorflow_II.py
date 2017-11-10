import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import urllib.request
from tensorflow.contrib import rnn
from custom_cell import CustomCell, LayerNormalizedLSTMCell
import reader
import collections

log_dir = "/home/rohola/tmp/rnn_basic"

###########create file#######
#with open('hello.txt', 'w') as fw:
#    for _ in range(1000000):
#        fw.write('hello ')
#############################


file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
#file_name = 'data/tinyshakespeare.txt'
file_name = 'data/howareyou.txt'
# if not os.path.exists(file_name):
#     urllib.request.urlretrieve(file_url, file_name)
#
# with open(file_name, 'r') as f:
#     raw_data = f.read()
#     print("Data length:", len(raw_data))
#
# raw_data = raw_data.strip('\n')
# vocab = set(raw_data)
# vocab_size = len(vocab)
# idx_to_vocab = dict(enumerate(vocab))
# vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
# data = [vocab_to_idx[c] for c in raw_data]
# del raw_data

##############Word level############################

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()

    content = [x.rstrip() for x in content]
    content = [x for x in content if x]
    content = [content[i].split() for i in range(len(content))]
    content = [item for sublist in content for item in sublist]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

training_data = read_data(file_name)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

vocab_to_idx, idx_to_vocab = build_dataset(training_data)
data = [vocab_to_idx[c] for c in training_data]
vocab_size = len(vocab_to_idx)




#################################################


def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)


def train_network(g, num_epochs, num_steps=200, batch_size=32, verbose=True, save=""):
    tf.set_random_seed(2345)

    with tf.Session() as sess:
        # tf.reset_default_graph()
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
                    training_loss, training_state, _ = sess.run([g['total_loss'],
                                                                 g['final_state'],
                                                                 g['train_step']],
                                                                feed_dict)
                    total_training_loss += training_loss
                    if verbose:
                        print("Average training loss for iteration", idx, " :", total_training_loss / steps)
                    #training_losses.append(total_training_loss / steps)

                        # if first:
                        #     X_1 = X_realized
                        #     first = False
                        #     continue
                        #
                        # if np.array_equal(X_realized, X_1):
                        #     print(idx)
                        #     break

                except:
                    print("exception occurred at "+str(idx))
                    continue

            #coord.request_stop()
            #coord.join()

            if save:
                g['saver'].save(sess, save)

        if save:
            print("saved")
            g['saver'].save(sess, save)

    return training_losses


def build_basic_rnn_graph_without_list(
        state_size=100,
        num_classes=vocab_size,
        batch_size=32,
        num_steps=200,
        learning_rate=1e-4):
    # reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)

    rnn_inputs = x_one_hot

    # cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    # cell = tf.nn.rnn_cell.LSTMCell(state_size)
    # cell = tf.nn.rnn_cell.GRUCell(state_size)
    #cell = CustomCell(state_size, num_weights=5)
    cell = LayerNormalizedLSTMCell(num_units=state_size)

    init_state = cell.zero_state(batch_size, tf.float32)

    output, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
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
    ################
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(output, W) + b
    logits = tf.reshape(logits, [batch_size, num_steps, num_classes])

    y_as_list = y  # [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

    loss_weights = tf.ones([batch_size, num_steps])  # [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.contrib.seq2seq.sequence_loss(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        saver=tf.train.Saver()
    )


def build_basic_rnn_graph_with_list(
        state_size=100,
        num_classes=vocab_size,
        batch_size=32,
        num_steps=200,
        learning_rate=1e-4):
    '''
    This function runs in legacy mode because instead of using tensors
    it uses the list of samples to calculate the loss function, this is so inefficient
    :param state_size:
    :param num_classes:
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

    #cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(state_size),rnn.BasicLSTMCell(state_size)])

    #init_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = rnn.static_rnn(cell, rnn_inputs, dtype=tf.float32)

    # outputs = []
    # state = init_state
    # with tf.variable_scope("RNN"):
    #     for time_step in range(num_steps):
    #         if time_step > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         (cell_output, state) = cell(rnn_inputs[time_step], state)
    #         outputs.append(cell_output)
    # final_state = state

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
        #init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds=predictions,
        saver=tf.train.Saver()
    )


def build_multilayer_lstm_graph_without_list(
        state_size=100,
        num_classes=vocab_size,
        batch_size=32,
        num_steps=200,
        num_layers=3,
        learning_rate=1e-4,
        build_with_dropout=False):
    # reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size], dtype=tf.float32)
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=0.7)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)

    init_state = cell.zero_state(batch_size, tf.float32)

    output, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    output = tf.reshape(output, [-1, state_size])
    ###########OR############
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
    #########################


    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(output, W) + b
    #logits = tf.reshape(logits, [batch_size, num_steps, num_classes])
    #logits = tf.reshape(logits, [batch_size, num_classes])

    predictions = tf.nn.softmax(logits)

    #loss_weights = tf.ones([batch_size, num_steps])

    y_reshaped = tf.reshape(y, [-1])

    #losses = tf.contrib.seq2seq.sequence_loss(logits, y, loss_weights)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )



def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'], g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]
                #current_char = np.argmax(np.squeeze(preds))

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return("".join(chars))


def generate_characters2(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_chars = [vocab_to_idx[c] for c in prompt]
        chars = current_chars


        for i in range(num_chars):
            current_chars = np.reshape(current_chars, (1, 10))
            # if state is not None:
            #     feed_dict = {g['x']: current_chars, g['init_state']: state}
            # else:
            feed_dict = {g['x']: current_chars}

            preds, state = sess.run([g['preds'], g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_chars = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                #current_chars = [np.random.choice(vocab_size, 1, p=np.squeeze(pred))[0] for pred in preds]
                current_chars = [np.argmax(np.squeeze(pred)) for pred in preds]

            chars+=current_chars

    chars = map(lambda x: idx_to_vocab[x], chars)
    print(" ".join(chars))
    return (" ".join(chars))

if __name__ == "__main__":
    num_steps = 10
    batch_size = 1
    # g = build_basic_rnn_graph_without_list()
    g = build_basic_rnn_graph_with_list(num_steps=num_steps, batch_size=batch_size)
    # g = build_multilayer_lstm_graph_without_list(num_steps=num_steps, batch_size=batch_size, num_layers=2, learning_rate=1e-4)
    #g = build_basic_rnn_graph_with_list()
    #train_network(g, num_epochs=100, save=log_dir, num_steps=num_steps, batch_size=batch_size)

    #input_ = "First Citizen: Before we proceed any further, hear me speak. All: Speak, speak. First Citizen: You are all resolved rather to die than to famish? All: Resolved. resolved."
    #input_ = "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello"
    input_ = ["how", "are", "you"]*10
    input_ = input_[0:10]
    text = generate_characters2(g, log_dir, 200, prompt=input_)
    print(text)
