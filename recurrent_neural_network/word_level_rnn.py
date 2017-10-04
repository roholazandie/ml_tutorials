import collections
import numpy as np
import reader
import tensorflow as tf
from tensorflow.contrib import rnn

log_dir = "/home/rohola/tmp/word_level_rnn"

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


def build_dataset(words):
    count = collections.Counter(words).most_common()
    vocab_to_idx = dict()
    for word, _ in count:
        vocab_to_idx[word] = len(vocab_to_idx)
    idx_to_vocab = dict(zip(vocab_to_idx.values(), vocab_to_idx.keys()))
    return vocab_to_idx, idx_to_vocab



def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)



def build_basic_rnn_graph_without_list(
        n_inputs,
        n_neurons=100,
        batch_size=32,
        n_steps=200,
        learning_rate=1e-4):
    '''
    the functions runs for each X input and then it gives a final state with outputs
    each X is composed of numsteps(timestep) matrix of batchsize x n_inputs

    :param n_inputs:
    :param n_neurons:
    :param batch_size:
    :param n_steps:
    :param learning_rate:
    :return:
    '''
    # reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, n_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, n_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, n_inputs)

    rnn_inputs = x_one_hot

    cell = tf.nn.rnn_cell.BasicRNNCell(n_neurons)
    # cell = tf.nn.rnn_cell.LSTMCell(state_size)
    # cell = tf.nn.rnn_cell.GRUCell(state_size)
    #cell = CustomCell(state_size, num_weights=5)
    #cell = LayerNormalizedLSTMCell(num_units=state_size)

    init_state = cell.zero_state(batch_size, tf.float32)

    output, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    output = tf.reshape(output, [-1, n_neurons])
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
        W = tf.get_variable('W', [n_neurons, n_inputs])
        b = tf.get_variable('b', [n_inputs], initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(output, W) + b
    logits = tf.reshape(logits, [batch_size, n_steps, n_inputs])

    predictions = tf.nn.softmax(logits)

    y_as_list = y  # [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

    loss_weights = tf.ones([batch_size, n_steps])  # [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.contrib.seq2seq.sequence_loss(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        predictions=predictions,
        train_step=train_step,
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
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        predictions=predictions,
        logits = logits,
        y_as_list = y_as_list,
        saver=tf.train.Saver()
    )


def train_network(g, num_epochs, num_steps=200, batch_size=32, verbose=True, save=""):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        total_training_loss = 0
        steps = 0
        training_state = None
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            for X, Y in epoch:
                try:
                    steps += 1

                    feed_dict = {g['x']: X, g['y']: Y}

                    if training_state is not None:
                       feed_dict[g['init_state']] = training_state


                    training_loss, training_state, _ = sess.run([g['total_loss'],
                                                                 g['final_state'],
                                                                 g['train_step']],
                                                                feed_dict)

                    #training_loss, _ , y_as_list, logits= sess.run([g['total_loss'], g['train_step'], g['y_as_list'], g['logits']], feed_dict)

                    #y_as_list = [item[0] for item in y_as_list]
                    #logits = [np.argmax(item[0]) for item in logits]
                    #print(logits, y_as_list)
                    total_training_loss += training_loss
                    if verbose:
                        print("Average training loss for iteration", idx, " :", total_training_loss / steps)

                except:
                    print("exception occurred at "+str(idx))
                    continue
            #if save:
            #g['saver'].save(sess, save)

        #if save:
        #print("saved")
        g['saver'].save(sess, save)
        #########################################
        num_words = 300
        input_words = ["how", "are", "you"] * 10
        input_words = input_words[0:10]

        current_words = [vocab_to_idx[c] for c in input_words]
        words = current_words

        for i in range(num_words):
            current_words = np.reshape(current_words, (1, 10))
            feed_dict = {g['x']: current_words}

            predictions, state = sess.run([g['predictions'], g['final_state']], feed_dict)

            # current_words = [np.random.choice(vocab_size, 1, p=np.squeeze(prediction))[0] for prediction in predictions]
            #current_words = [np.argmax(np.squeeze(prediction)) for prediction in predictions]

            current_words = np.argmax(predictions,axis=2)
            #words += current_words
            words+=list(current_words[0])

        words = map(lambda x: idx_to_vocab[x], words)
        print(" ".join(words))

    return training_losses


def generate_words(g, checkpoint, num_words, input_words=[], pick_top_words=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'] = tf.train.import_meta_graph(checkpoint+'.meta')
        g['saver'].restore(sess, checkpoint)

        current_words = [vocab_to_idx[c] for c in input_words]
        words = current_words


        for i in range(num_words):
            current_words = np.reshape(current_words, (1, 10))
            feed_dict = {g['x']: current_words}

            predictions, state = sess.run([g['predictions'], g['final_state']], feed_dict)

            if pick_top_words is not None:
                p = np.squeeze(predictions)
                p[np.argsort(p)[:-pick_top_words]] = 0
                p = p / np.sum(p)
                current_words = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                #current_words = [np.random.choice(vocab_size, 1, p=np.squeeze(prediction))[0] for prediction in preds]
                current_words = [np.argmax(np.squeeze(prediction)) for prediction in predictions]

            words+=current_words

    words = map(lambda x: idx_to_vocab[x], words)
    #print(" ".join(words))
    return (" ".join(words))


if __name__ == "__main__":
    file_name = 'data/howareyou.txt'
    #file_name = 'data/tinyshakespeare.txt'
    training_data = read_data(file_name)
    vocab_to_idx, idx_to_vocab = build_dataset(training_data)
    data = [vocab_to_idx[c] for c in training_data]
    vocab_size = len(vocab_to_idx)

    mode = 'train'
    #mode = 'test'
    num_steps = 10
    batch_size = 1

    #g = build_basic_rnn_graph_with_list(num_classes=vocab_size, num_steps=num_steps, batch_size=batch_size)
    g = build_basic_rnn_graph_without_list(n_inputs=vocab_size, n_steps=num_steps, batch_size=batch_size)
    if mode == "train":
        train_network(g, num_epochs=100, save=log_dir, num_steps=num_steps, batch_size=batch_size)
    else:
        input_words = ["how", "are", "you"] * 10
        input_words = input_words[0:10]
        text = generate_words(g, log_dir, 200, input_words=input_words)
        print(text)

