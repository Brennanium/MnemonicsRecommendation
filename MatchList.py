import aline
import nltk
from math import trunc
from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial import distance

phonetic_multiplier = 5.0
orthographic_multiplier = 3
semantic_multiplier = 50
aoa_multiplier = 3.0


model = KeyedVectors.load_word2vec_format("word_embeddings/english/glove.6B.50d.txt", binary=False)
def semantic_distance(word1, word2, multiplier=1.0):
    try:
        word1_embedding = model[word1]
        word2_embedding = model[word2]
        dist = distance.cosine(word1_embedding, word2_embedding) * multiplier
        return dist
    except:
        return 20 * multiplier



class Match:
    def __init__(self, input_node, translation):
        self.matched_words = ''
        self.matched_phones = ''
        self.target_word = input_node.word
        self.target_phones = input_node.phones
        self.target_definitions = input_node.definitions
        self.translation = translation
        self.delta = 0
        self.is_finished = False
        self.search_failed = False # in case final phones can't be matched
        self.unmatched_phones = input_node.phones

    def is_fully_matched(self):
        return self.is_finished


    def mark_search_failed(self):
        self.search_failed = True
        self.is_finished = True

    def get_phones_unmatched(self):
        if self.matched_phones == '':
            return self.target_phones

        alignment = aline.align(self.matched_phones, self.target_phones)

        # check for end of alignment
        len_alignment = len(alignment[0])
        last_idx_aligned = len_alignment
        for i in reversed(range(len(alignment[0]))):
            if alignment[0][i][0] != '-':
                last_idx_aligned = i + 1
                break

        if last_idx_aligned == len_alignment:
            self.is_finished = True
            # add on the orthographc distance to the delta once all phones are matched
            self.delta += nltk.edit_distance(self.matched_words, self.target_word) * orthographic_multiplier
            return None
        else:
            return self.target_phones[last_idx_aligned:]

    def add_new_matched_phones(self, node, phonetic_delta):
        self.matched_words = self.matched_words + ' ' + node.word
        self.matched_phones = self.matched_phones + node.phones
        self.delta += phonetic_delta * phonetic_multiplier
        self.delta += semantic_distance(node.word, self.translation, semantic_multiplier)
        self.delta += node.aoa * aoa_multiplier
        self.unmatched_phones = self.get_phones_unmatched()
        if not self.unmatched_phones:
            self.is_finished = True
            self.matched_words = self.matched_words.strip()

class MatchList:
    def __init__(self):
        self.unfinished_matches = []
        self.finished_matches = []

    def print_matches(self):
        print("Unfinished Matches:")
        if self.unfinished_matches:
            for match in sorted(self.unfinished_matches, key=lambda x: x.delta):
                match.print_match()
        print("Finished Matches:")
        if self.finished_matches:
            for match in sorted(self.finished_matches, key=lambda x: x.delta):
                match.print_match()

    def add_match(self, match):
        if match.get_phones_unmatched() == None:
            self.finished_matches.append(match)
        else:
            self.unfinished_matches.append(match)

    def remove_and_retrieve_unfinished_matches(self, N=10):
        temp_unfinished = []
        for match in sorted(self.unfinished_matches, key=lambda x: x.delta)[:N]:
            if match.is_fully_matched() and not match.search_failed:
                self.finished_matches.append(match)
            else:
                temp_unfinished.append(match)
        self.unfinished_matches = []
        return temp_unfinished

    def get_finished_matches(self, N=10):
        return sorted(self.finished_matches, key=lambda x: x.delta)[:N]
