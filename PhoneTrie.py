import csv
import re
import aline
import heapq
import math

non_speech_marks = ['ˈ', '.', "'" ,'ˌ' ,'ː', '̯', ':', '-', '-', '|', '|', '/', 'ᵝ',
                    'ᵊ', '\u200b', 'ʲ', '˞', '̈', '—', 'ʰ', '̶', ',', '̆', '\xa0',
                    '\u2009', 'ˑ',  '·', 'ʷ', '̥', 'ˡ', '`', '̝', '̙', '/', '\\',
                    '̃', '(', ')', '̩', '͡', '\u200c', '(', '̪', '̚', 'ᵊ', ' ']
brackets = ['[',']','(',')','\{','\}','/','|','\\']
invalid_symbols = {'.'}

def is_valid_word(word):
    for c in word:
        if c not in aline.feature_matrix:
            invalid_symbols.add(c)
            return False
    return True

# def remove_stress_marks(input):
#     phones = input
#     for c in non_speech_marks:
#         phones = phones.replace(c, '')
#     return phones

# def remove_brackets(input):
#     phones = input
#     for c in brackets:
#         phones = phones.replace(c, '')
#     return phones

class PhoneNode:


    def __init__(self, char):
        """
        Stores data for a Node in the PhoneTrie
        """
        self.char = char
        self.is_word = False
        self.word = ""
        self.phones = ""
        self.phones_raw = ""
        self.children = {}
        self.aoa = 0
        self.ignored = False
        self.definitions = None
        self.next = None # used for when words collide, aka 2 words same pronunciation


    def __lt__(self, other):
        """
        This is necessiary because nodes get added to a heap to compare how similar
        they are to another node as a tuple (delta, node), if 2 deltas are the same
        the heap implementation looks to the next value in the tuple which is the node
        to choose which is most similar
        """
        return 0

class PhoneTrie:

    def __init__(self, language_code):
        """
        Initialize a trie where each node PhoneNode, trie is built from input dictionaries that store
        the word, the words phones, and the age of aquisition for the word.
        Words with the same phones/prounciation are chained using the PhoneTrie.next field

        :param language_code: language to build trie from {'de, 'fr', 'en', 'zh', 'ja'}
        :raises   ValueError: raises value error when provided language_code not supported
        """
        self.root = PhoneNode("")
        self.num_nodes = 0
        self.n_best_list = [] # heap
        if language_code == 'de': # german
            self.insert_dictionary('dictionaries/german/phones_de.csv')
        elif language_code == 'en': # english
            self.insert_dictionary('dictionaries/english/top_8000_words.csv')
        elif language_code == 'fr': # french
            self.insert_dictionary('dictionaries/french/fr_FR.csv')
        elif language_code == 'zh': # mandarin
            self.insert_dictionary('dictionaries/chinese/zh.csv')
        elif language_code == 'ja': # japanese
            self.insert_dictionary('dictionaries/japanese/phones_ja.csv')
        else:
            raise ValueError("Language " + language_code + " is not supported. Currently languages de, fr, zh, and ja are supported.")

    def is_valid_entry(entry):
        """
        Return if a entry is composed only of ipa symbols used in the aline algorithm

        :param   entry: string
        :returns: true if entry is composed of only ipa symbols used in the aline algorithm
        """
        for c in entry:
            if c not in aline.feature_matrix:
                return False
        return True

    def remove_stress_marks(input):
        """
        Remove all characters not used by the aline algorithm and return

        :param input: string to have characters removed from
        :returns: input string with any characters not used in the aline algorithm removed
        """
        for c in input:
            if c not in aline.feature_matrix.keys():
                input = input.replace(c, '')
        return input

    def remove_brackets(input):
        """
        Remove all characters not used by the aline algorithm and return

        :param input: string to have characters removed from
        :returns: input string with any characters not used in the aline algorithm removed
        """
        phones = input
        for c in brackets:
            phones = phones.replace(c, '')
        return phones

    # Inserts a word and it's phones into the trie, if two words have
    # the same pronunciation/phones, nodes will be strung into a list
    # starting with the first word with the pronunciation entered.
    # Subsequent nodes with the same pronunciation can be found using
    # node.next values
    def insert(self, word, phones, phones_raw, aoa):
        """
        Insert a new node into the trie. If two words have the same pronunciation, they will
        arrive at the same node, if this happens the new node gets saved in the node.next
        field

        :param   word: word to insert into trie
        :param phones: phones for word being inserted
        :param    aoa: int, age of aquisition for word being entered
        """
        if ' ' in phones or ' ' in word or not PhoneTrie.is_valid_entry(phones):
            return

        node = self.root
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
        node.phones_raw = phones_raw
        node.aoa = aoa
        self.num_nodes = self.num_nodes + 1


    def search(self, word):
        """
        Return the node from the trie that corresponds to word
        """
        return self.__search(self.root, word)

    def __search(self, node, word):
        """
        Return the node from the trie that corresponds to word
        :param node: node to be checked, and all children checked
        :param word: word being searched for
        :returns: None if no word found, otherwise the node containing word
        """
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
        """
        Writes the trie to a file in the same format as the dictionaries
        a dictionary line looks like:
        word ; pronunciation, pronunciation ; age of aquisition
        """
        with open(file_path, 'w+') as new_file:
            self.__write_trie_to_file(self.root, new_file)


    def __write_trie_to_file(self, node, open_file):
        """
        Writes a node to the open file in the style
        word ; pronunciation, pronunciation ; age of aquisition
        Then calls __write_trie_to_file on all nodes children
        """
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

    def insert_dictionary(self, path_to_dict):
        """
        Inserts entries from a dictionary file into the trie
        """
        with open(path_to_dict, mode = 'r') as dataFile:
            for row in dataFile:
                cols = row.split(';')
                for pronunciation in cols[1].split(','):
                    self.insert(cols[0], PhoneTrie.remove_stress_marks(pronunciation), PhoneTrie.remove_brackets(pronunciation), float(cols[2]))



    def __add_to_running_list(self, delta_and_node):
        """
        For use with find_phonetic_match, adds a node and it's delta to the running
        list of N best phonetic matches.

        The N best list is done as a heap so the max delta node can be removed
        when the heap has more than N nodes
        Also updates the self.max value so the search can abandon early words when
        the delta exceeds the max

        :param delta_and_node: (delta, node), a tuple containing the nodes delta and the node
        """
        if not delta_and_node[1].ignored:
            if len(self.n_best_list) < self.N:
                heapq.heappush(self.n_best_list, delta_and_node)
            else:
                self.max = heapq.heappushpop(self.n_best_list, delta_and_node)[0]

    def find_phonetic_match(self, unfinished_match, N):
        """
        Searches the trie for a similar set of phones to the unmatched phones of unfinished_match
        Similarity is done by comparing an unmatched phone with a phone from the trie and adding
        their similarity score to a running total delta, the similarity score is found using
        the aline algorithms' delta() function. This allows for reusing delta for words with
        common prefixes.

        :param unfinished_match: a Match object containing unmatched phones to be matched
        :param                N: number of matches to return
        :returns: a list of tuples containing (delta, node) where delta is the totat delta
                    accumulated finding the match and node stores the word/phones for the match
        """
        self.n_best_list = []
        self.N = N
        self.max = -math.inf
        for c in self.root.children:
            self.__find_phonetic_match(self.root.children[c], unfinished_match.unmatched_phones, 0)
        return self.n_best_list

    def __find_phonetic_match(self, node, phones, phonetic_delta):
        """
        Recursively add all possible matches from the trie to the running N best match list,
        updating the running phonetic delta along the way. Abandon early if match is more
        dissimilar than all current matches in the list, as their children will be also

        :param           node: Current node being added, and whose children will be added
        :param         phones: string of phones yet paired
        :param phonetic_delta: phonetic delta for the nodes prefix
        """
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

