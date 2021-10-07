import unittest
import DictionaryTrie as dt

class TestMatchMethods(unittest.TestCase):

    def test_is_valid_word(self):
        word = "-word"
        self.assertFalse(dt.is_valid_word(word))

    def test_invalid_symbol_remove(self):
        word = "testˈ.ˌː,word"
        self.assertEqual("testword", dt.remove_stress_marks(word))

    def test_trie_search(self):
        # test word retrieval
        en_trie = dt.DictTrie("en")
        en_trie.insert("fakephones", "fakeword")
        node = en_trie.search("fakeword")
        self.assertEqual(node.word, "fakeword")
        self.assertEqual(node.phones, "fakephones")

        # test finding non-existant node
        false_node = en_trie.search("anotherfakeword")
        self.assertEqual(None, false_node)
