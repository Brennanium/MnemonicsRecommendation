import csv
import re
import aline
import heapq
import math

non_speech_marks = ['ˈ', '.', "'" ,'ˌ' ,'ː', '̯', ':', '-', '-', '|', '|', '/', 'ᵝ'
                    'ᵊ', '\u200b', 'ʲ', '˞', '̈', '—', 'ʰ', '̶', ',', '̆', '\xa0',
                    '\u2009', 'ˑ',  '·', 'ʷ', '̥', 'ˡ', '`', '̝', '̙', '/', '\\',
                    '̃', '(', ')', '̩', '͡', '\u200c', '(', '̪', '̚', 'ᵊ', ' ']

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

class PhoneNode:
    def __init__(self, char):
        self.char = char
        self.is_word = False
        self.word = ""
        self.phones = ""
        self.children = {}
        self.aoa = 0
        self.ignored = False
        self.definitions = None
        self.next = None # used for when words collide, aka 2 words same pronunciation

    # Necessiary to have this for when nodes get sorted in the heap because if two
    # nodes have the same delta the heap will check the second value in the tuple
    # which is the node
    def __lt__(self, other):
        return 0

class PhoneTrie:
    def __init__(self, language_code):
        self.root = PhoneNode("")
        self.num_nodes = 0
        self.depth = 0
        self.lookup = {}
        self.n_best_list = [] # heap
        if language_code == 'de': # german
            self.insert_dictionary('dictionaries/german/phones_de.csv')
        elif language_code == 'en': # english
            self.insert_dictionary('dictionaries/english/phones_and_definitions_en_US.csv')
        elif language_code == 'fr': # french
            self.insert_dictionary()
        elif language_code == 'zh': # mandarin
            self.insert_dictionary()
        elif language_code == 'ja': # japanese
            self.insert_dictionary('dictionaries/japanese/phones_ja.csv')
        elif language_code == 'test':
            self.insert_dictionary('dictionaries/english/phones_test_US.csv')
        else:
            raise Exception("Language " + language_code + " is not supported. Currently languages de, fr, zh, and ja are supported.")

    # Inserts a word and it's phones into the trie, if two words have
    # the same pronunciation/phones, nodes will be strung into a list
    # starting with the first word with the pronunciation entered.
    # Subsequent nodes with the same pronunciation can be found using
    # node.next values
    def insert(self, word, phones, aoa, definitions):
        if ' ' in phones or ' ' in word or not is_valid_word(phones):
            return
        node = self.root
        if len(phones) > self.depth:
            self.depth = len(phones)

        for char in phones:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = PhoneNode(char)
                node.children[char] = new_node
                node = new_node

        if node.is_word: # collision, already had a word with same prounciation
            while node.next != None:
                node = node.next
            node.next = PhoneNode(node.char)
            node = node.next

        node.is_word = True
        node.word = word.lower()
        node.phones = phones
        node.definitions = definitions
        node.aoa = aoa
        self.num_nodes = self.num_nodes + 1


    def search(self, word):
        return self.__search(self.root, word)

    def __search(self, node, word):
        if node.is_word:
            try:
                node.aoa = self.lookup[node.word]
            except:
                node.aoa = 25
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

    def write_trie_to_file(self, file_path):
        new_file = open(file_path, 'w+')
        self.__write_trie_to_file(self.root, new_file)
        new_file.close()

    def __write_trie_to_file(self, node, open_file):
        if node.is_word:
            open_file.write(node.word + ";" + node.phones + ";" + str(node.aoa))
            if node.definitions:
                open_file.write(";"+node.definitions[0])
                for d in node.definitions[1:]:
                    open_file.write(";" + d)
            open_file.write("\n")
        for c in node.children: # check all nodes children
            child_node = node.children[c]
            found_node = self.__write_trie_to_file(child_node, open_file)
        while node.next != None: # check all different words with same pronunciation
            node = node.next
            self.__write_trie_to_file(node, open_file)

    # takes a path to a csv formated file where in the first
    # column there is the foreign word, and the second is the
    # phones for the word in the first column
    # inserts all words into the trie
    def insert_dictionary(self, path_to_csv_dict):
        with open(path_to_csv_dict, mode = 'r') as dataFile:
            for row in dataFile:
                parts = row[:-1].split(';')
                for pronunciation in parts[1].split(','):
                    definitions = None
                    if len(parts) > 3:
                        definitions = parts[3:]
                    self.insert(parts[0], remove_stress_marks(pronunciation), int(parts[2]), definitions)

    # won't necessiarly print in what you would consider alphabetical order
    def print_trie(self):
        self.__print_trie_recurse(self.root)

    def __print_trie_recurse(self, node):
        if node.is_word:
            print(node.word, node.phones, node.definitions)
        for c in node.children:
            self.__print_trie_recurse(node.children[c])

    # Adds a trie node to the running list stored as a min heap, where nodes
    # are entered based on their -delta, so that the largest delta get removed
    # first. Only will hold up to N elements, also updates the current max delta
    # so that potential matches that exceed that delta can be abandoned before
    # they complete their whole search
    def __add_to_running_list(self, delta_and_node):
        if not delta_and_node[1].ignored:
            if len(self.n_best_list) < self.N:
                heapq.heappush(self.n_best_list, delta_and_node)
            else:
                self.max = heapq.heappushpop(self.n_best_list, delta_and_node)[0]

    # searces the trie for phoneticly similar matches to the unmatched phones
    # held by unfinished_match
    def find_phonetic_match(self, unfinished_match, N):
        self.n_best_list = []
        self.N = N
        self.max = -math.inf
        for c in self.root.children:
            self.__find_phonetic_match(self.root.children[c], unfinished_match.unmatched_phones, 0)
        return self.n_best_list

    def __find_phonetic_match(self, node, phones, phonetic_delta):
        if not phones:
            return
        phonetic_delta = phonetic_delta - aline.delta(phones[0], node.char)
        if phonetic_delta > self.max: # stop searching if all children will we worse than existing matches
            if node.is_word:
                temp = node
                while temp: # add all the words with the same pronuciation
                    self.__add_to_running_list((phonetic_delta, temp))
                    temp = temp.next

            for c in node.children:
                self.__find_phonetic_match(node.children[c], phones[1:], phonetic_delta)
















a = 0
