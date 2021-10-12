from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial import distance
import numpy as np
model = KeyedVectors.load_word2vec_format("word_embeddings/glove.6B.50d.txt", binary=False)

def get_distance(word1, word2, multiplier=1.0):
    word1_embedding = model[word1]
    word2_embedding = model[word2]
    return distance.cosine(word1_embedding, word2_embedding) * multiplier
