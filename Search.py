import subprocess
from DictionaryTrie import DictTrie
import DictionaryTrie as dt
import aline
import MatchList
import copy
from datetime import datetime

supported_languages = {'en', 'ja', 'de', 'fr'}

class WWUTransphoner:

    def __init__(self, input_language, target_language):
        if input_language and target_language not in supported_languages:
            raise Exception("Must choose from supported languages:", supported_languages)
        else:
            self.target_trie = DictTrie(target_language)
            self.input_trie = DictTrie(input_language)

    def __get_input_word_phones(self, input_word):
        input_node = self.input_trie.search(input_word)
        if input_node:
            return input_node.phones
        else:
            raise Exception("Can't find phones for input word:", input_word)

    def get_mnemonics(self, input_word, N=5):
        input_phones = self.__get_input_word_phones(input_word)
        print("Found phones for word:", input_phones)
        matches = self.target_trie.find_phonetic_match(input_phones, N)
        match_list = MatchList.MatchList()
        for delta, node in matches[:N]:
            match = MatchList.Match(node.word, node.phones,
                                      input_word, input_phones, -delta, node.aoa)
            match_list.add_match(match)

        working_matches = match_list.remove_and_retrieve_unfinished_matches(N)
        while working_matches:
            for match in working_matches:
                unmatched_phones = match.get_phones_unmatched()
                if unmatched_phones: # sometimes a match will be fully matched after the initial guess
                    potential_matches = self.target_trie.find_phonetic_match(unmatched_phones, N)
                    if potential_matches: # sometimes a match wont find anything to match
                        for i in range(0, min(N, len(potential_matches))):
                            new_match = copy.deepcopy(match)
                            new_match.add_new_matched_phones(potential_matches[i][1], -potential_matches[i][0])
                            match_list.add_match(new_match)
                    else:
                        match.mark_search_failed()
                        match_list.add_match(match)
                working_matches = match_list.remove_and_retrieve_unfinished_matches(N)

        return match_list.get_finished_matches(N)

start = datetime.now()
wwwt = WWUTransphoner('ja', 'en')
print("Transphoner Loaded")
end = datetime.now()
print("first phase:", end - start)
matches = wwwt.get_mnemonics("準備完了ログ記録")
end2 = datetime.now()
print("second phase:", end2 - end)
for m in matches:
    print(m.matched_words)
