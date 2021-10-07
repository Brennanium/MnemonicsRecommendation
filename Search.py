import subprocess
from DictionaryTrie import DictTrie
import DictionaryTrie as dt
import numpy as np
import aline
import MatchList
import copy

N = 10 # number of best matches to keep track of

input_language = 'de'
target_language = 'en'
input_word = 'misstraut' # german for 'Assumption of Risk'

en_trie = DictTrie('en')
de_trie = DictTrie('de')

# Converts the input word to it's phonetic symbols
input_phones = ""
input_node = de_trie.search(input_word)
if input_node:
    input_phones = input_node.phones
    print(input_word + "'s phones:", input_phones)
else:
    raise Exception("Can't find phones for input word:", input_word)

# Gets a list of matches sorted by smallest distance
# from input to matched word ascending
matches = en_trie.find_phonetic_match(input_phones)
matches.sort(key=lambda x:x[1])

match_list = MatchList.MatchList()

for node, delta in matches[:N]:
    match = MatchList.Match(node.word, node.phones,
                          input_word, input_phones, delta)
    match_list.add_match(match)

working_matches = match_list.remove_and_retrieve_unfinished_matches(N)
while working_matches:
    for match in working_matches:
        unmatched_phones = match.get_phones_unmatched()
        if unmatched_phones: # sometimes a match will be fully matched after the initial guess
            potential_matches = en_trie.find_phonetic_match(unmatched_phones)
            if potential_matches: # sometimes a match wont find anything to match
                potential_matches.sort(key=lambda x:x[1])
                for i in range(0, min(N, len(potential_matches))):
                    new_match = copy.deepcopy(match)
                    new_match.add_new_matched_phones(potential_matches[i][0], potential_matches[i][1])
                    match_list.add_match(new_match)
            else:
                match.mark_search_failed()
                match_list.add_match(match)
        working_matches = match_list.remove_and_retrieve_unfinished_matches(N)

match_list.print_matches()





















abcd=4
