import tensorflow as tf
import os

from tensorflow.contrib.data.python.ops.dataset_ops import Dataset
from tensorflow.python.ops.tensor_array_ops import TensorArray


def simple_file_reading():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/olympics2016.csv"

    features = tf.placeholder(tf.int32, shape=[3], name='features')
    country = tf.placeholder(tf.string, name='country')
    total = tf.reduce_sum(features, name='total')
    # tf.Print basically logs the current values in the second parameter
    # (in this case, the list [country, features, total]) and returns the first value (total).
    printerop = tf.Print(total, [country, features, total], name='printer')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with open(filename) as inf:
            # Skip header
            next(inf)
            for line in inf:
                # Read data, using python, into our features
                country_name, code, gold, silver, bronze, total = line.strip().split(",")
                gold = int(gold)
                silver = int(silver)
                bronze = int(bronze)
                # Run the Print ob
                total = sess.run(printerop, feed_dict={features: [gold, silver, bronze], country: country_name})
                print(country_name, total)


def read_csv_file_tensorflow(filenames):


    def create_file_reader_ops(filename_queue):
        reader = tf.TextLineReader(skip_header_lines=1)
        _, csv_row = reader.read(filename_queue)
        record_defaults = [[""], [""], [0], [0], [0], [0]]
        country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
        features = tf.stack([gold, silver, bronze])
        return features, country

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    features, country = create_file_reader_ops(filename_queue)


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        # The reason is that the queue itself doesn’t sit on the graph in the same way a normal operation does,
        # so we need a Coordinator to manage running through the queue.
        # This co-ordinator will increment through the dataset everytime example and label are evaluated, as they effectively pull data from the file.
        coord = tf.train.Coordinator()
        # You must call tf.train.start_queue_runners
        # to populate the queue before you call run or eval to execute the read
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            try:
                features_values, country_name = sess.run([features, country])
                print(features_values, country_name)
            except tf.errors.OutOfRangeError:
                break

        coord.request_stop()
        coord.join(threads)


def batching(filenames):
    def create_file_reader_ops(filename_queue):
        reader = tf.TextLineReader(skip_header_lines=1)
        _, csv_row = reader.read(filename_queue)
        record_defaults = [[""], [""], [0], [0], [0], [0]]
        country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
        features = tf.stack([gold, silver, bronze])
        return features, country

    def input_pipeline(filenames, batch_size, num_epochs=None):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
        example, label = create_file_reader_ops(filename_queue)
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return example_batch, label_batch


    example_batch, label_batch = input_pipeline(filenames=filenames, batch_size=10, num_epochs=4)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            try:
                features_values, country_names = sess.run([example_batch, label_batch])
                print(features_values, country_names)
            except tf.errors.OutOfRangeError:
                break

        coord.request_stop()
        coord.join(threads)


def multi_batching(filenames):
    '''
    the same as above(i.e. batching) but with multiple batches and more parallelism
    '''
    def create_file_reader_ops(filename_queue):
        reader = tf.TextLineReader(skip_header_lines=1)
        _, csv_row = reader.read(filename_queue)
        record_defaults = [[""], [""], [0], [0], [0], [0]]
        country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
        features = tf.stack([gold, silver, bronze])
        return features, country

    def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
        example_list = [create_file_reader_ops(filename_queue) for _ in range(read_threads)]
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        example_batch, label_batch = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return example_batch, label_batch

    example_batch, label_batch = input_pipeline(filenames=filenames, read_threads=7, batch_size=10, num_epochs=4)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            try:
                features_values, country_names = sess.run([example_batch, label_batch])
                print(features_values, country_names)
            except tf.errors.OutOfRangeError:
                break

        coord.request_stop()
        coord.join(threads)



def dataset_reading(filenames):
    dataset = Dataset.from_tensor_slices(filenames)
    print(dataset.output_types)
    print(dataset.output_shapes)

    dataset = dataset.flat_map(
        lambda filename: (
            tf.contrib.data.TextLineDataset(filename)
                .skip(1)
                .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                value = sess.run(next_element)
                print(value)
            except tf.errors.OutOfRangeError:
                break

if __name__ == "__main__":
    #simple_file_reading()
    #read_csv_file_tensorflow(filenames=["olympics2016.csv"])
    #batching(filenames=["olympics2016.csv"])
    #multi_batching(filenames=["olympics2016.csv"])
    dataset_reading(filenames=["olympics2016.csv"])