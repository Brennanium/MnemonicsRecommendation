import aline
import nltk

phonetic_multiplier = 1.0
orthographic_multiplier = 1.5
aoa_multiplier = 1.0

class Match:
    def __init__(self, target_word, target_phones):
        self.matched_words = ''
        self.matched_phones = ''
        self.target_word = target_word
        self.target_phones = target_phones
        self.delta = 0
        self.is_finished = False
        self.search_failed = False # in case final phones can't be matched
        self.unmatched_phones = self.get_phones_unmatched()

    def is_fully_matched(self):
        return self.is_finished

    def mark_search_failed(self):
        self.search_failed = True
        self.is_finished = True

    def print_match(self):
        print("Match: ", self.target_phones, "|",
                self.matched_phones, "|", self.matched_words, "|", self.delta)


    def get_phones_unmatched(self):
        if self.matched_phones == '':
            return self.target_phones

        alignment = aline.align(self.matched_phones, self.target_phones)

        # check for end of alignment
        len_alignment = len(alignment[0])
        last_idx_aligned = len_alignment
        for i in reversed(range(len(alignment[0]))):
            if alignment[0][i][0] != '-':
                last_idx_aligned = i + 1
                break

        if last_idx_aligned == len_alignment:
            self.is_finished = True
            return None
        else:

            return self.target_phones[last_idx_aligned:]

    def add_new_matched_phones(self, node, delta):
        self.matched_words = self.matched_words + ' ' + node.word
        self.matched_phones = self.matched_phones + node.phones
        self.delta += delta
        self.unmatched_phones = self.get_phones_unmatched()
        if not self.unmatched_phones:
            self.is_finished = True

class MatchList:
    def __init__(self):
        self.unfinished_matches = []
        self.finished_matches = []

    def print_matches(self):
        print("Unfinished Matches:")
        if self.unfinished_matches:
            for match in sorted(self.unfinished_matches, key=lambda x: x.delta):
                match.print_match()
        print("Finished Matches:")
        if self.finished_matches:
            for match in sorted(self.finished_matches, key=lambda x: x.delta):
                match.print_match()

    def add_match(self, match):
        if match.get_phones_unmatched() == None:
            self.finished_matches.append(match)
        else:
            self.unfinished_matches.append(match)

    def remove_and_retrieve_unfinished_matches(self, N=10):
        temp_unfinished = []
        for match in sorted(self.unfinished_matches, key=lambda x: x.delta)[:N]:
            if match.is_fully_matched():
                self.finished_matches.append(match)
            else:
                temp_unfinished.append(match)
        self.unfinished_matches = []
        return temp_unfinished

    def get_finished_matches(self, N=10):
        return self.finished_matches[:min(N, len(self.finished_matches))]
