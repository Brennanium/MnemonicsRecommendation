import csv
import re
import aline
import heapq
import math

# age of aquisition
aoa_multiplier = 0

non_speech_marks = ['ˈ', '.', "'" ,'ˌ' ,'ː', '̯', ':', '-', '-', '|', '|', '/', 'ᵝ'
                    'ᵊ', '\u200b', 'ʲ', '˞', '̈', '—', 'ʰ', '̶', ',', '̆', '\xa0',
                    '\u2009', 'ˑ',  '·', 'ʷ', '̥', 'ˡ', '`', '̝', '̙', '/', '\\',
                    '̃', '(', ')', '̩', '͡', '\u200c', '(', '̪', '̚', 'ᵊ']

invalid_symbols = {'.'}

def is_valid_word(word):
    for c in word:
        if c not in aline.feature_matrix:
            invalid_symbols.add(c)
            return False
    return True

def remove_stress_marks(input):
    phones = input
    for c in non_speech_marks:
        phones = phones.replace(c, '')
    return phones

class TrieNode:
    def __init__(self, char):
        self.char = char
        self.is_word = False
        self.word = ""
        self.phones = ""
        self.children = {}
        self.aoa = 0.0
        self.has_aoa = False
        self.next = None # used for when words collide, aka 2 words same pronunciation

    def __lt__(self, other):
        return 0

    def set_aoa(self, aoa):
        self.aoa = aoa
        self.has_aoa = True

class DictTrie:
    def __init__(self, language_code):
        self.root = TrieNode("")
        self.num_nodes = 0
        self.depth = 0
        self.n_best_list = [] # heap
        if language_code == 'de': # german
            self.insert_dictionary('dictionaries/german/de.csv')
        elif language_code == 'en': # english
            self.insert_dictionary('dictionaries/english/en_US.csv')
            #self.add_aoa_ratings('dictionaries/english/AoA_ratings_english.csv')
        elif language_code == 'fr': # french
            self.insert_dictionary()
        elif language_code == 'zh': # mandarin
            self.insert_dictionary()
        elif language_code == 'ja': # japanese
            self.insert_dictionary('dictionaries/japanese/ja.csv')
        else:
            raise Exception("Language " + language_code + " is not supported. Currently languages de, fr, zh, and ja are supported.")
        print(invalid_symbols)

    def insert(self, phones, word):
        if ' ' in phones or ' ' in word or not is_valid_word(phones):
            return
        node = self.root
        if len(phones) > self.depth:
            self.depth = len(phones)

        for char in phones:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node

        if not node.is_word:
            node.is_word = True
            node.word = word
            node.phones = phones
        else: # collision, already had a word with same prounciation
            while node.next != None:
                node = node.next
            node.next = TrieNode(node.char)
            node.next.is_word = True
            node.next.word = word
            node.next.phones = phones
            node.next.next = None


    def search(self, word):
        return self.__search(self.root, word)

    def __search(self, node, word):
        if node.is_word and node.word == word:
            return node
        else:
            for c in node.children: # check all nodes children
                child_node = node.children[c]
                found_node = self.__search(child_node, word)
                if found_node != None:
                    return found_node
            while node.next != None: # check all different words with same pronunciation
                node = node.next
                if node.is_word and node.word == word:
                    return node
            return None


    # takes a path to a csv formated file where in the first
    # column there is the foreign word, and the second is the
    # phones for the word in the first column
    # inserts all words into the trie
    def insert_dictionary(self, path_to_csv_dict):
        with open(path_to_csv_dict, mode = 'r') as dataFile:
            file_reader = csv.reader(dataFile, delimiter=",")
            found = False
            for row in file_reader:
                phones = remove_stress_marks(row[1])
                self.insert(phones, row[0])
                self.num_nodes = self.num_nodes + 1

    def add_aoa_ratings(self, path_to_aoa_data):
        with open(path_to_aoa_data, mode = 'r') as dataFile:
            file_reader = csv.reader(dataFile, delimiter="\t")
            for row in file_reader:
                found_node = self.search(row[0])
                if found_node:
                    found_node.set_aoa(float(row[1]) * aoa_multiplier)

    # won't necessiarly print in what you would consider alphabetical order
    def print_trie(self):
        self.__print_trie_recurse(self.root)

    def __print_trie_recurse(self, node):
        if node.is_word:
            print(node.word)
        for c in node.children:
            self.__print_trie_recurse(node.children[c])

    def __add_to_running_list(self, delta_and_node):
        if len(self.n_best_list) < self.N:
            heapq.heappush(self.n_best_list, delta_and_node)
        else:
            self.min = heapq.heappushpop(self.n_best_list, delta_and_node)[0]

    def find_phonetic_match(self, input_phones, N):
        self.n_best_list = []
        self.N = N
        self.min = -math.inf
        for c in self.root.children:
            self.__find_phonetic_match(self.root.children[c], input_phones, 0)
        return self.n_best_list

    def __find_phonetic_match(self, node, phones, delta):
        if not phones:
            return
        delta = delta - aline.delta(phones[0], node.char)
        if delta > self.min: # stop searching if all children will we worse than existing matches
            if node.is_word:
                if node.has_aoa:
                    self.__add_to_running_list((delta + node.aoa, node))
                else:
                    self.__add_to_running_list((delta, node))

            for c in node.children:
                self.__find_phonetic_match(node.children[c], phones[1:], delta)
















a = 0
