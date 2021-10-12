import unittest
import DictionaryTrie as dt

class TestDictTrieMethods(unittest.TestCase):

    def test_is_valid_word(self):
        word = "-word"
        self.assertFalse(dt.is_valid_word(word))

    def test_invalid_symbol_remove(self):
        word = "testˈ.ˌː,word"
        self.assertEqual("testword", dt.remove_stress_marks(word))

    def test_trie_search(self):
        en_trie = dt.DictTrie("en")

        # test word retrieval on word with 1 pronunciation
        node = en_trie.search('thaxter')
        self.assertEqual(node.word, "thaxter")
        self.assertEqual(node.phones, "θækstɝ")

        # test word retrieval on word with multiple pronciations
        node = en_trie.search('the')
        self.assertEqual(node.word, 'the')

        # test finding non-existant node
        false_node = en_trie.search("fakeword")
        self.assertEqual(None, false_node)

        false_node = en_trie.search('')
        self.assertEqual(None, false_node)

    def test_trie_insert(self):
        en_trie = dt.DictTrie('en')
        en_trie.insert("fakephones", "fakeword")
        node = en_trie.search("fakeword")
        self.assertEqual(node.word, 'fakeword')
        self.assertEqual(node.phones, 'fakephones')

    def test_small_dict(self):
        test_trie = dt.DictTrie('test')
        self.assertEqual(test_trie.num_nodes, 12)
