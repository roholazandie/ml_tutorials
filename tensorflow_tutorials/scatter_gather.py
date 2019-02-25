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


if __name__ == "__main__":
    '''
    Whereas in tf.gather indices defines slices into the first dimension of params, 
    in tf.gather_nd, indices defines slices into the first N dimensions of params, where N = indices.shape[-1].

    '''
    pad_mask = tf.constant([0, 1, 0, 1, 1, 0, 0, 1], dtype=tf.float32)
    nonpad_mask = 1 - pad_mask
    padded_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
    non_padded_ids = tf.to_int32(tf.where(nonpad_mask < 1e-9))
    with tf.Session() as sess:
        print(sess.run(padded_ids))

    data = tf.constant([0, 11, 0, 10, 9, 0, 0, 12])
    gathered = gathering(data, non_padded_ids)

    scattering(gathered, non_padded_ids, tf.constant([8]))