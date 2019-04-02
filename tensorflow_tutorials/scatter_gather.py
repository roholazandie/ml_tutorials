import tensorflow as tf



def scattering(updates, indices, shape):
    # indices = tf.constant([[4], [3], [1], [7]])
    # updates = tf.constant([9, 10, 11, 12])
    # shape = tf.constant([8])
    scattered = tf.scatter_nd(indices, updates, shape)

    with tf.Session() as sess:
        scattered_value = sess.run(scattered)
        print(scattered_value)

    return scattered_value


def gathering(updates, indices):
    #indices = tf.constant([[4], [3], [1], [7]])
    #updates = tf.constant([9, 10, 11, 12])
    gathered = tf.gather_nd(updates, indices)

    with tf.Session() as sess:
        gathered_value = sess.run(gathered)
        print(gathered_value)

    return gathered_value


def gather1():
    positions = tf.constant([2,4], tf.int32)
    seq_length = 6
    batch_size = 5
    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape( positions + flat_offsets, [-1])
    print("====")
    with tf.Session() as sess:
        print(sess.run(flat_offsets))
        print(sess.run(flat_positions))



if __name__ == "__main__":

    pad_mask = tf.constant([0, 1, 0, 1, 1, 0, 0, 1], dtype=tf.float32)
    nonpad_mask = 1 - pad_mask # [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    padded_ids = tf.to_int32(tf.where(pad_mask < 1e-9)) #[0, 2, 5, 6]
    non_padded_ids = tf.to_int32(tf.where(nonpad_mask < 1e-9)) #[1, 3, 4, 7]
    with tf.Session() as sess:
        print(sess.run(padded_ids))
        print(sess.run(non_padded_ids))

    data = tf.constant([0, 11, 0, 10, 9, 0, 0, 12])
    gathered = gathering(data, non_padded_ids)

    scattering(gathered, non_padded_ids, tf.constant([8]))


    data = tf.constant([['a', 'b'], ['c', 'd']], dtype=tf.string)
    indices = [[0, 0], [1, 1]]
    gathering(data, indices)

    gather1()