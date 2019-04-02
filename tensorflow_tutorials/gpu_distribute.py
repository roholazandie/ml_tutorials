import tensorflow as tf
import numpy as np
import time

BATCH_SIZE = 512
EPOCHS = 5
NUM_GPUS = 4


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)
        loss_value = run_values.results
        print("loss value:", loss_value)


def input_fn(images, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 5000
    ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    ds = ds.prefetch(2)
    # Return the dataset. (L)
    return ds


def architecture(inputs, mode, scope='MnistConvNet'):
    """Return the output operation following the network architecture.
    Args:
        inputs (Tensor): Input Tensor
        mode (ModeKeys): Runtime mode (train, eval, predict)
        scope (str): Name of the scope of the architecture
    Returns:
         Logits output Op for the network.
    """
    with tf.variable_scope(scope):
        inputs = inputs / 255
        input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=20,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        flatten = tf.reshape(pool2, [-1, 4 * 4 * 40])
        dense1 = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense1, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout, units=10)
        return dense2

def model_fn(mode, features, labels, params):
    logits = architecture(features, mode)
    class_predictions = tf.argmax(logits, axis=-1)
    predictions = class_predictions

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())


        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions
        )



if __name__ == "__main__":
    # the data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    TRAINING_SIZE = len(train_images)
    TEST_SIZE = len(test_images)
    train_images = np.asarray(train_images, dtype=np.float32) / 255
    # Convert the train images and add channels
    train_images = train_images.reshape((TRAINING_SIZE, 28, 28, 1))
    test_images = np.asarray(test_images, dtype=np.float32) / 255
    # Convert the test images and add channels
    test_images = test_images.reshape((TEST_SIZE, 28, 28, 1))
    # Cast the labels to floats, needed later
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)


    distribution = tf.contrib.distribute.MirroredStrategy()

    run_config = tf.estimator.RunConfig(train_distribute=distribution)

    time_hist = TimeHistory()

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                        config=run_config)
    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: input_fn(train_images,
    #                                      train_labels,
    #                                      epochs=1,
    #                                      batch_size=BATCH_SIZE))
    #
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: input_fn(test_images,
    #                                      test_labels,
    #                                      epochs=1,
    #                                      batch_size=BATCH_SIZE),
    #     steps=None,
    #     start_delay_secs=10,  # Start evaluating after 10 sec.
    #     throttle_secs=30  # Evaluate only every 30 sec
    # )
    #
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec, hooks=[time_hist])

    estimator.train(input_fn=lambda: input_fn(train_images,
                                         train_labels,
                                         epochs=1,
                                         batch_size=BATCH_SIZE), hooks=[time_hist])



    # estimator.evaluate(lambda: input_fn(test_images,
    #                                     test_labels,
    #                                     epochs=1,
    #                                     batch_size=BATCH_SIZE))

    total_time = sum(time_hist.times)
    print("total time with"+ str(NUM_GPUS) + "GPU(s):"+str(total_time)+"seconds")

    avg_time_per_batch = np.mean(time_hist.times)
    print(str(BATCH_SIZE * NUM_GPUS / avg_time_per_batch)+" images/second with+"+str(NUM_GPUS)+"GPU(s)")