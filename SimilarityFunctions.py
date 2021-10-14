from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial import distance
model = KeyedVectors.load_word2vec_format("word_embeddings/english/glove.6B.50d.txt", binary=False)

aoa_lookup = {}
file = open('word_embeddings/english/en_aoa_ratings.csv')
for line in file:
    chunks = line.split(',')
    aoa_lookup[chunks[0]] = chunks[1]

def get_distance(word1, word2, multiplier=1.0):
    try:
        word1_embedding = model[word1]
        word2_embedding = model[word2]
        dist = distance.cosine(word1_embedding, word2_embedding) * multiplier
        return dist
    except:
        return 20 * multiplier

def get_aoa(word, aoa_multiplier=2.0):
    try:
        return aoa_lookup[word] * aoa_multiplier
    except:
        return 25 * aoa_multiplier