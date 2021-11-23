from PhoneTrie import PhoneTrie
import MatchList
import copy
from gensim.models import KeyedVectors
from gensim import models
from datetime import datetime
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from happytransformer import HappyWordPrediction
import random
from datetime import datetime

supported_languages = {'en', 'ja', 'de', 'fr', 'zh'}

punctuation = ['.', '!', '?']

class WWUTransphoner:

    def __init__(self, input_language):
        print("Creating", input_language, "trie.")
        if input_language not in supported_languages:
            raise ValueError("\"" + input_language + "\"", "not in supported languages:", supported_languages)
        else:
            self.target_trie = PhoneTrie('en')
            self.input_trie = PhoneTrie(input_language)
            self.happy_wp = HappyWordPrediction()
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
            self.input_language = input_language


    def set_multipliers(self, imageability=1.0, orthographic=1, phonetic=1.0, semantic=50):
        MatchList.aoa_multiplier = imageability
        MatchList.phonetic_multiplier = phonetic
        MatchList.semantic_multiplier = semantic
        MatchList.orthographic_multiplier = orthographic

    def get_mnemonics(self, input_word, translation, N=5, include_phones=False):
        # if N is too small the closest phonetic match may be
        # nonsense, and it won't allow for the other metrics to
        # replace it
        old_N = N
        N = max(N, 5)

        input_node = self.input_trie.search(input_word.lower())
        if not input_node:
            #raise Exception("Can't find phones for input word:", input_word)
            print("Couldn't find phones for input word:", input_word)
            return None

        starting_match = MatchList.Match(input_node, translation)
        match_list = MatchList.MatchList()
        match_list.add_match(starting_match)

        working_matches = match_list.remove_and_retrieve_unfinished_matches(N)
        while working_matches:
            for match in working_matches:
                potential_matches = self.target_trie.find_phonetic_match(match, N)
                if potential_matches: # sometimes a match wont find anything to match
                    for i in range(0, min(N, len(potential_matches))):
                        new_match = copy.deepcopy(match)
                        new_match.add_new_matched_phones(potential_matches[i][1], -potential_matches[i][0])
                        match_list.add_match(new_match)
                else:
                    match.mark_search_failed()
                    match_list.add_match(match)
            working_matches = match_list.remove_and_retrieve_unfinished_matches(N)
        
        if include_phones:
            words = [ match.matched_words for match in match_list.get_finished_matches(old_N) ]
            phones = [ "/" + match.matched_phones_raw.strip() + "/" for match in match_list.get_finished_matches(old_N) ]
            return words, phones, "/" + input_node.phones_raw + "/"
        else:
            return [ match.matched_words for match in match_list.get_finished_matches(old_N) ]

    # Marks a word in the target trie as unusuable, so Subsequent
    # mnemonics will not contain it again
    def mark_ignored(self, word):
        node = self.target_trie.search(word)
        if node:
            node.ignored = True

    def gen_sentence(self, input_mnemonics):

        sentences = []

        # Generates the end of the sentence for each input mnemonic
        inputs = []
        for i in input_mnemonics:
            inputs.append(self.tokenizer.encode(i, return_tensors='pt')[0])

        # Input mnemonics get encoded into different token lengths, and the model,
        # needs equal length inputs for all inputs in a batch so we seperate them out
        # into batches of equal token length to speed up the generation
        token_lenghts = {len(encoding) for encoding in inputs}
        for lenght in token_lenghts:
            curr_list = [encoding for encoding in inputs if len(encoding) == lenght]
            outputs = self.model.generate(torch.stack(curr_list), max_length=15, do_sample=True)
            for tensor in outputs:
                sentences.append(self.tokenizer.decode(tensor, skip_special_tokens=True))

        # Generates the start of each sentence before the mnemonic
        for i in range(len(sentences)):

            # Cuts down the output from the model to just one sentence
            last_idx = len(sentences[i])
            for p in punctuation:
                try:
                    last_idx = min(last_idx, sentences[i].index(p))
                except:
                    pass
            sentences[i] = sentences[i][:last_idx+1]

            last_token = ''
            for _ in range(random.randrange(0,8)):
                temp = '[MASK] ' + sentences[i]
                result = self.happy_wp.predict_mask(temp, top_k=5)
                new_token = result[random.randrange(0, 5)].token
                attemps = 0
                while (new_token == '•' or last_token == new_token) and attemps < 5:
                    new_token = result[random.randrange(0, 5)].token
                    last_token = new_token
                    attemps += 1
                sentences[i] = new_token + ' ' + sentences[i]

        return sentences
