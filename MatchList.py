from math import trunc
import nltk
from scipy.spatial import distance
from gensim.models.keyedvectors import KeyedVectors
import aline

phonetic_multiplier = 5.0
orthographic_multiplier = 3
semantic_multiplier = 50
aoa_multiplier = 3.0

model = KeyedVectors.load_word2vec_format("word_embeddings/english/glove.6B.50d.txt", binary=False)
def semantic_distance(word1, word2, multiplier=1.0):
    """
    Return the cosine distance between the word2vec embeddings for word1, word2.
    If no embeddings are found, defaults to 20. Currently only has values for english.
    """
    try:
        word1_embedding = model[word1]
        word2_embedding = model[word2]
        dist = distance.cosine(word1_embedding, word2_embedding) * multiplier
        return dist
    except:
        return 20 * multiplier

class Match:


    def __init__(self, input_node, translation=None):
        """
        The Match class can be used to store data about a mnemonic match as it is built.

        :param   input_node: node in a PhoneTrie to build a Match out of
        :param  translation: translation of word in output language, used for semantic difference
        """
        self.matched_words = ''
        self.matched_phones = ''
        self.matched_phones_raw = ''
        self.target_word = input_node.word
        self.target_phones = input_node.phones
        self.target_phones_raw = input_node.phones_raw
        self.target_definitions = input_node.definitions
        self.translation = translation
        self.delta = 0
        self.is_fully_matched = False
        self.search_failed = False # in case final phones can't be matched
        self.unmatched_phones = input_node.phones

    def get_phones_unmatched(self):
        """
        Return the yet unmatched target phones. Phones are determined yet unmatched if they
        proceed the last matched phone after using the ALINE alignment algorithm.

        Example using letters instead of phones:
        the alinement for 'watermelon' and 'water':
            w a t e r m e l o n
            w a t e r - - - - -
        characters after the 'r' in watermelon are considered unmatched

        :returns: the unmatched phones after aligning the matched phones and target phones
        """

        if self.matched_phones == '':
            return self.target_phones

        # Get alignment using the ALINE algorithm
        alignment = aline.align(self.matched_phones, self.target_phones)

        # Find end of alignment
        len_alignment = len(alignment[0])
        last_idx_aligned = len_alignment
        for i in reversed(range(len(alignment[0]))):
            if alignment[0][i][0] != '-':
                last_idx_aligned = i + 1
                break

        if last_idx_aligned == len_alignment:
            self.is_fully_matched = True
            # add on the orthographc distance to the delta once all phones are matched
            self.delta += nltk.edit_distance(self.matched_words, self.target_word) * orthographic_multiplier
            return None
        else:
            return self.target_phones[last_idx_aligned:]

    def add_new_matched_phones(self, node, phonetic_delta):
        """
        Update the match with the newly matched phones.

        :param           node: the PhoneTrie node for the matched word/phones
        :param phonetic_delta: the phonetic distance between the matched phones and the node

        """
        self.matched_words = self.matched_words + ' ' + node.word
        self.matched_phones = self.matched_phones + node.phones
        self.matched_phones_raw = self.matched_phones_raw + ' ' + node.phones_raw

        self.delta += phonetic_delta * phonetic_multiplier
        self.delta += node.aoa * aoa_multiplier
        if self.translation:
            self.delta += semantic_distance(node.word, self.translation, semantic_multiplier)

        self.unmatched_phones = self.get_phones_unmatched()
        if not self.unmatched_phones:
            self.is_fully_matched = True
            self.matched_words = self.matched_words.strip()

class MatchList:


    def __init__(self):
        """
        MatchList is used to keep track of the matches as they are updated.
        """
        self.unfinished_matches = []
        self.finished_matches = []

    def add_match(self, match):
        """
        Add the match to the MatchList

        :param match: the match to be added
        """
        if match.get_phones_unmatched() == None:
            self.finished_matches.append(match)
        else:
            self.unfinished_matches.append(match)

    def remove_and_retrieve_unfinished_matches(self, N=10):
        """
        Return the N best unfinished matches, and clear the unfinished_matches list
        Checks the matches are not complete first.

        :param N: max number of matches to retrieve (default 10)
        :returns: returns the top N best incomplete matches
        """
        temp_unfinished = []
        for match in sorted(self.unfinished_matches, key=lambda x: x.delta)[:N]:
            if match.is_fully_matched and not match.search_failed:
                self.finished_matches.append(match)
            else:
                temp_unfinished.append(match)
        self.unfinished_matches = []
        return temp_unfinished

    def get_finished_matches(self, N=10):
        """
        Return the top N best finished matches

        :param N: max number of matches to retrieve (default 10)
        :returns: the top N completed matches
        """
        return sorted(self.finished_matches, key=lambda x: x.delta)[:N]
