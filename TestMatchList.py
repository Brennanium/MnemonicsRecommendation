import unittest
import MatchList

class TestMatchMethods(unittest.TestCase):

    def test_match(self):
        test_match = MatchList.Match("word", "wɝd", "word", "wɝd", 0)
        unmatched_phones = test_match.get_phones_unmatched()
        self.assertEqual(None, unmatched_phones)
        self.assertTrue(test_match.is_fully_matched())

    def test_matchlist_add_remove(self):
        test_match_finished = MatchList.Match("graceful", "ɡɹeɪsfʊl", "graceful", "ɡɹeɪsfʊl", 0)
        test_match_unfinished = MatchList.Match("gray", "ɡɹeɪ", "graceful", "ɡɹeɪsfʊl", 0)
        match_list = MatchList.MatchList()
        match_list.add_match(test_match_finished)
        match_list.add_match(test_match_unfinished)
        retrieved_matches = match_list.remove_and_retrieve_unfinished_matches()
        finished_matches = match_list.get_finished_matches()
        self.assertTrue(len(retrieved_matches) == 1)
        self.assertTrue(retrieved_matches[0].unmatched_phones == 'sfʊl')
        self.assertTrue(len(finished_matches) == 1)
        self.assertTrue(finished_matches[0].matched_phones, "ɡɹeɪsfʊl")
