import csv
import re
import aline

aoa_multiplier = 0

non_speech_marks = ['ˈ', '.', "'" ,'ˌ' ,'ː', '̯', ':', '-', '-', '|', '|',
                    'ᵊ', '\u200b', 'ʲ', '˞', '̈', '—', 'ʰ', '̶', ',', '̆', '\xa0',
                    '\u2009', 'ˑ',  '·', 'ʷ', '̥', 'ˡ', '`', '̝', '̙',
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

    def set_aoa(self, aoa):
        self.aoa = aoa
        self.has_aoa = True

class DictTrie:
    def __init__(self, language_code):
        self.root = TrieNode("")
        self.num_nodes = 0
        self.depth = 0
        if language_code == 'de': # german
            self.insert_dictionary('dictionaries/german/german-dict-with-phones.csv')
        elif language_code == 'en': # english
            self.insert_dictionary('dictionaries/english/lemmas_no_hyphens_with_ipa.csv')
            #self.add_aoa_ratings('dictionaries/english/AoA_ratings_english.csv')
            #self.insert_dictionary('dictionaries/english/english-with-phones.csv')
        #elif language_code == 'fr': # french
        #    self.insert_dictionary()
        #elif language_code == 'zh': # chinease
        #    self.insert_dictionary()
        #elif language_code == 'ja': # japanese
        #    self.insert_dictionary()
        else:
            raise Exception("Language " + language_code + " is not supported. Currently languages de, fr, zh, and ja are supported.")

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
        node.is_word = True
        node.word = word
        node.phones = phones

    def search(self, word):
        return self.__search(self.root, word)

    def __search(self, node, word):
        if node.is_word and node.word == word:
            return node
        else:
            for c in node.children:
                child_node = node.children[c]
                found_node = self.__search(child_node, word)
                if found_node != None:
                    return found_node
            return None


    # takes a path to a csv formated file where in the first
    # column there is the foreign word, and the second is the
    # phones for the word in the first column
    # inserts all words into the trie
    def insert_dictionary(self, path_to_csv_dict):
        with open(path_to_csv_dict, mode = 'r') as dataFile:
            file_reader = csv.reader(dataFile, delimiter="\t")
            for row in file_reader:
                if (len(row[0]) != 1) and row[0][0] != '-': # no 1 letter dict words or prefixes
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

    def find_phonetic_match(self, input_phones):
        final_list = []
        for c in self.root.children:
            final_list.extend(self.__find_phonetic_match(self.root.children[c], input_phones, 0))
        return final_list

    def __find_phonetic_match(self, node, phones, delta):
        if not phones:
            return None

        delta = delta + aline.delta(phones[0], node.char)
        running_list = []
        if node.is_word:
            if node.has_aoa:
                running_list.append((node, delta + node.aoa))
            else:
                running_list.append((node, delta))

        for c in node.children:
            new_list = self.__find_phonetic_match(node.children[c], phones[1:], delta)
            if new_list != None:
                running_list.extend(new_list)

        return running_list
















a = 0
