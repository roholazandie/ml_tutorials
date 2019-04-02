import tensorflow as tf


def core_model(input_img, num_classes=10):
    """
        A simple model to perform classification on 28x28 grayscale images in MNIST style.

        Args:
        input_img:  A floating point tensor with a shape that is reshapable to batchsizex28x28. It
            represents the inputs to the model
        num_classes:  The number of classes
    """
    net = tf.reshape(input_img, [-1, 28, 28, 1])
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu,
                           name="conv2d_1")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu,
                           name="conv2d_2")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.reshape(net, [-1, 7 * 7 * 64])
    net = tf.layers.dense(inputs=net, units=1024, name="dense_1", activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=net, units=num_classes, name="dense_2")
    return logits


def training_model(input_fn):
    inputs = input_fn()
    image = inputs[0]
    label = tf.cast(inputs[1], tf.int32)
    logits = core_model(image)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
    return tf.reduce_mean(loss)



def training_dataset(epochs=5, batch_size=128):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets("data")
    all_data_points = mnist_data.train.next_batch(60000)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(all_data_points)
    dataset = dataset.repeat(epochs).shuffle(10000).batch(batch_size)
    return dataset


def do_training(update_op, loss):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            step = 0
            while True:
                _, loss_value = sess.run((update_op, loss))
                if step % 100 == 0:
                    print('Step {} with loss {}'.format(step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final loss: {}'.format(loss_value))


def serial_training(model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()

    loss = model_fn(lambda: iterator.get_next())
    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    global_step = tf.train.get_or_create_global_step()
    update_op = optimizer.minimize(loss, global_step=global_step)

    do_training(update_op, loss)


tf.reset_default_graph()
serial_training(training_model, training_dataset(epochs=2))


def parallel_training(model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    update_op, loss = create_parallel_optimization(model_fn,
                                                   input_fn,
                                                   optimizer)

    do_training(update_op, loss)


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


def create_parallel_optimization(model_fn, input_fn, optimizer, controller="/cpu:0"):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()

    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                # Compute loss and gradients, but don't apply them yet
                loss = model_fn(input_fn)

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)

    return apply_gradient_op, avg_loss


def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



if __name__ == "__main__":
    tf.reset_default_graph()
    parallel_training(training_model, training_dataset(epochs=2))