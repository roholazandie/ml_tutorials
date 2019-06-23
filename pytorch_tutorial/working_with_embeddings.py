from torch import nn
import torch
import gensim
import numpy as np

def word2vec_from_pytorch():
    model = gensim.models.KeyedVectors.load_word2vec_format("/home/rohola/data/GoogleNews-vectors-negative300.bin",
                                                            binary=True)
    man_vec = model["man"]
    woman_vec = model["woman"]
    glass_vec = model["glass"]

    print(np.dot(man_vec, woman_vec))
    print(np.dot(man_vec, glass_vec))

    weights = torch.FloatTensor(model.vectors)

    embedding = nn.Embedding.from_pretrained(weights)

    input = torch.LongTensor([1])
    word_vector = embedding(input)

    print(word_vector)


def random_embedding():
    embedding = nn.Embedding(num_embeddings=10, embedding_dim=300)

    input = torch.LongTensor([1])
    word_vector = embedding(input)

    print(word_vector)


    mat_input = torch.LongTensor([[1, 2], [3, 1]])

    word_vectors = embedding(mat_input)
    print(word_vectors)


if __name__ == "__main__":
    #word2vec_from_pytorch()
    random_embedding()