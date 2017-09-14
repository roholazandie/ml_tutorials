import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _linear


class CustomCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, num_weights):
        self._num_units = num_units
        self._num_weights = num_weights


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                ru = _linear([inputs, state], 2 * self._num_units, bias=False)
                ru = tf.nn.sigmoid(ru)
                #r, u = tf.split(1, 2, ru)
                r, u = tf.split(ru, 2, axis=1)
            with tf.variable_scope("Candidate"):
                lambdas = _linear([inputs, state], self._num_weights, True)
                #lambdas = tf.split(1, self._num_weights, tf.nn.softmax(lambdas))
                lambdas = tf.split(tf.nn.softmax(lambdas), self._num_weights, axis=1)

                Ws = tf.get_variable("Ws", shape = [self._num_weights, inputs.get_shape()[1], self._num_units])
                #Ws = [tf.squeeze(i) for i in tf.split(0, self._num_weights, Ws)]
                Ws = [tf.squeeze(i) for i in tf.split(Ws, self._num_weights, axis=0)]

                candidate_inputs = []

                for idx, W in enumerate(Ws):
                    candidate_inputs.append(tf.matmul(inputs, W) * lambdas[idx])


                Wx = tf.add_n(candidate_inputs)
                with tf.variable_scope("C"):
                    c = tf.nn.tanh(Wx + _linear([r * state],
                                                self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h], 4 * self._num_units, False)

            i, j, f, o = tf.split(concat, 4, axis=1)

            # add layer normalization to each gate
            i = self.ln(i, scope = 'i/')
            j = self.ln(j, scope = 'j/')
            f = self.ln(f, scope = 'f/')
            o = self.ln(o, scope = 'o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(self.ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


    def ln(self, tensor, scope=None, epsilon=1e-5):
        """ Layer normalizes a 2D tensor along its second axis """
        assert (len(tensor.get_shape()) == 2)
        m, v = tf.nn.moments(tensor, [1], keep_dims=True)
        if not isinstance(scope, str):
            scope = ''
        with tf.variable_scope(scope + 'layer_norm'):
            scale = tf.get_variable('scale',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(1))
            shift = tf.get_variable('shift',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(0))
        LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

        return LN_initial * scale + shift