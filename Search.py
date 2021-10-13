from PhoneTrie import PhoneTrie
import MatchList
import copy
import tracemalloc
from gensim.models import KeyedVectors
from gensim import models
supported_languages = {'en', 'ja', 'de', 'fr'}

class WWUTransphoner:

    def __init__(self, input_language):
        if input_language not in supported_languages:
            raise Exception("Must choose from supported languages:", supported_languages)
        else:
            self.target_trie = PhoneTrie('en')
            self.input_trie = PhoneTrie(input_language)

    def __get_input_word_phones(self, input_word):
        input_node = self.input_trie.search(input_word)
        if input_node:
            return input_node.phones
        else:
            raise Exception("Can't find phones for input word:", input_word)

    def get_mnemonics(self, input_word, translation, N=5):
        input_node = self.input_trie.search(input_word)
        print("Found phones for word:", input_node.phones)
        starting_match = MatchList.Match(input_node, translation)
        match_list = MatchList.MatchList()
        match_list.add_match(starting_match)

        working_matches = match_list.remove_and_retrieve_unfinished_matches(N)
        while working_matches:
            for match in working_matches:
                potential_matches = self.target_trie.find_phonetic_match(match, N)
                if potential_matches: # sometimes a match wont find anything to match
                    for i in range(0, min(N, len(potential_matches))):
                        new_match = copy.deepcopy(match)
                        new_match.add_new_matched_phones(potential_matches[i][1], -potential_matches[i][0])
                        match_list.add_match(new_match)
                else:
                    match.mark_search_failed()
                    match_list.add_match(match)
            working_matches = match_list.remove_and_retrieve_unfinished_matches(N)

        for m in match_list.get_finished_matches(N):
            m.print_match()
        return match_list.get_finished_matches(N)

    # Marks a word in the target trie as unusuable, so Subsequent
    # mnemonics will not contain it again
    def mark_ignored(self, word):
        node = self.target_trie.search(word)
        if node:
            node.ignored = True


wwwt = WWUTransphoner('de')
matches = wwwt.get_mnemonics("tropisch", "tropical", 10)
