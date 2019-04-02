import tensorflow as tf
import numpy as np
import codecs

def get_iterator(hyper_parameters,
                 source_vocab_table,
                 target_vocab_table,
                 source_dataset,
                 target_dataset):

    batch_size = hyper_parameters["batch_size"]
    eos = hyper_parameters["eos"]
    sos = hyper_parameters["sos"]
    source_max_len = hyper_parameters["source_max_len"]
    target_max_len = hyper_parameters["target_max_len"]

    output_buffer_size = batch_size*1# * 1000

    source_eos_id = tf.cast(source_vocab_table.lookup(tf.constant(eos)), tf.int32)

    target_sos_id = tf.cast(target_vocab_table.lookup(tf.constant(sos)), tf.int32)
    target_eos_id = tf.cast(target_vocab_table.lookup(tf.constant(eos)), tf.int32)

    source_target_dataset = tf.data.Dataset.zip((source_dataset, target_dataset))

    # source_target_dataset = source_target_dataset.shuffle(buffer_size=output_buffer_size,
    #                                                       reshuffle_each_iteration=True)

    source_target_dataset = source_target_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values)).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    source_target_dataset = source_target_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if source_max_len:
        source_target_dataset = source_target_dataset.map(
            lambda src, tgt: (src[:source_max_len], tgt)).prefetch(output_buffer_size)

    if target_max_len:
        source_target_dataset = source_target_dataset.map(
            lambda src, tgt: (src, tgt[:target_max_len])).prefetch(output_buffer_size)

    source_target_dataset = source_target_dataset.map(
        lambda src, tgt: (tf.cast(source_vocab_table.lookup(src), tf.int32),
                          tf.cast(target_vocab_table.lookup(tgt), tf.int32))).prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    source_target_dataset = source_target_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([target_sos_id], tgt), 0),
                          tf.concat((tgt, [target_eos_id]), 0))).prefetch(output_buffer_size)

    source_target_dataset = source_target_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(output_buffer_size)

    source_target_dataset = source_target_dataset.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            source_eos_id,  # src
            target_eos_id,  # tgt_input
            target_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

    iterator = source_target_dataset.make_initializable_iterator()

    (source_ids, target_input_ids, target_output_ids, source_sequence_lenght, target_sequence_length) = iterator.get_next()
    return (iterator, source_ids, target_input_ids, target_output_ids, source_sequence_lenght, target_sequence_length)


def load_embed_txt(embed_file):
  """Load embed_file into a python dictionary.

  Note: the embed_file should be a Glove/word2vec formatted txt file. Assuming
  Here is an example assuming embed_size=5:

  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547

  For word2vec format, the first line will be: <num_words> <embedding_size>.

  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
  embedding_dict = dict()
  embedding_size = None

  is_first_line = True
  with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, "rb")) as f:
    for line in f:
      tokens = line.rstrip().split(" ")
      if is_first_line:
        is_first_line = False
        if len(tokens) == 2:  # header line
          embedding_size = int(tokens[1])
          continue
      word = tokens[0]
      vector = list(map(float, tokens[1:]))
      embedding_dict[word] = vector
      if embedding_size:
        if embedding_size != len(vector):
          del embedding_dict[word]
      else:
        embedding_size = len(vector)
  return embedding_dict, embedding_size


def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size

def create_pretrained_emb_from_txt(vocab_file, embedding_file, num_trainable_tokens=3, dtype=tf.float32):
    """Load pretrain embeding from embed_file, and return an embedding matrix.

      Args:
        embed_file: Path to a Glove formated embedding txt file.
        num_trainable_tokens: Make the first n tokens in the vocab file as trainable
          variables. Default is 3, which is "<unk>", "<s>" and "</s>".
      """
    vocab, _ = load_vocab(vocab_file)
    trainable_tokens = vocab[:num_trainable_tokens]
    embedding_dict, embedding_size = load_embed_txt(embedding_file)

    for token in trainable_tokens:
        if token not in embedding_dict:
            embedding_dict[token] = [0.0] * embedding_size

    embedding_matrix = np.array([embedding_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype)
    #todo what's the point of adding trainable tokens when we slice them out
    const_part_embedding_matrix = tf.slice(tf.constant(embedding_matrix), [num_trainable_tokens, 0], [-1, -1])
    var_part_embedding_matrix = tf.get_variable("var_part_embedding_matrix",shape = [num_trainable_tokens, embedding_size])
    embedding_matrix = tf.concat([var_part_embedding_matrix, const_part_embedding_matrix], 0)
    return embedding_matrix


def create_or_load_embeddings(embed_name,
                              vocab_file,
                              embedding_file,
                              vocab_size,
                              embed_size,
                              dtype):
    if vocab_file and embedding_file: #using pretrained embeddings
        embedding = create_pretrained_emb_from_txt(vocab_file, embedding_file)
    else: #create embedding to be learned in the end to end syste,
        embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)
    return embedding


def test(sequence):
    source_vocab_file = "text_data/source_vocab.txt"
    source_embedding_file = "text_data/source_embedding.txt"
    embedding = create_or_load_embeddings("myembedding",
                                          vocab_file=source_vocab_file,
                                          embedding_file=source_embedding_file,
                                          vocab_size=10,
                                          embed_size=4,
                                          dtype=tf.float32)
    sequence = [0]

    input_embedding = tf.nn.embedding_lookup(embedding, ids=sequence)
    print(input_embedding)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(sequence.initializer)
        val = sess.run(input_embedding)
        print(val)

if __name__ == "__main__":
    #tf.enable_eager_execution()
    #test(None)

    hyper_parameters = {}
    hyper_parameters["batch_size"] = 1
    hyper_parameters["sos"] = "<s>"
    hyper_parameters["eos"] = "</s>"
    hyper_parameters["source_max_len"] = 20
    hyper_parameters["target_max_len"] = 10

    source_vocab_file = "text_data/source_vocab.txt"
    target_vocab_file = "text_data/target_vocab.txt"
    train_source_file = "text_data/train_source_file.txt"
    train_target_file = "text_data/train_target_file.txt"
    UNK_ID = 0


    source_vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=source_vocab_file,
                                                                 default_value=UNK_ID)

    target_vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=target_vocab_file,
                                                                 default_value=UNK_ID)

    source_dataset = tf.data.TextLineDataset(tf.gfile.Glob(train_source_file))
    target_dataset = tf.data.TextLineDataset(tf.gfile.Glob(train_target_file))

    (iterator, source_ids, target_input_ids, target_output_ids, source_seq_len, target_seq_len) = get_iterator(hyper_parameters,
                 source_vocab_table,
                 target_vocab_table,
                 source_dataset,
                 target_dataset)


    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.tables_initializer())
    #     sess.run(iterator.initializer)
    #
    #     try:
    #         # Keep running next_batch till the Dataset is exhausted
    #         while True:
    #             a = sess.run(target_input_ids)
    #             print(a)
    #     except tf.errors.OutOfRangeError:
    #         print("out of range")
    #         pass


    source_embedding_file = "text_data/source_embedding.txt"
    embedding = create_or_load_embeddings("myembedding",
                                          vocab_file=source_vocab_file,
                                          embedding_file=source_embedding_file,
                                          vocab_size=10,
                                          embed_size=3,
                                          dtype=tf.float32)


    #source_ids = source_vocab_table.lookup(tf.constant("word2", dtype=tf.string))
    source_input_embedding = tf.nn.embedding_lookup(embedding, ids=source_ids)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)

        print(sess.run(source_input_embedding))

        # embed, ids = sess.run((source_input_embedding, source_ids))
        # print(embed, ids)

        # idx = sess.run(source_ids)
        # print(idx)

