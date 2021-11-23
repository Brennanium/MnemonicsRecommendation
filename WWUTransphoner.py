import random
import copy
from datetime import datetime
from gensim import models
from gensim.models import KeyedVectors
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from happytransformer import HappyWordPrediction
import MatchList
from PhoneTrie import PhoneTrie


class WWUTransphoner:

    supported_languages = {'en', 'ja', 'de', 'fr', 'zh'}

    def __init__(self, input_language, output_language):
        """
        The WWUTransphoner provides the functionality of taking in a word in one language,
        and outputting a short mnemonic phrase for use in second language vocabulary memorization
        The mnemonic phrase is similar to the input word by 4 metrics:
            - semantic meaning between the two
            - orthographic distance between the two (Levenshtein distance)
            - phonetic difference between the two
            - the average year people aquire the word (age of aquisition)

        It also provides the functionality of turning those short mnemonic phrases,
        or phrases manually created by the user into full complete sentences

        :param  input_language: the language correspoding to the potential input words
        :param output_language: language for outputing mnemonics in
        :raises     ValueError: raises value error when provided input/output language not supported
        """

        if input_language not in WWUTransphoner.supported_languages:
            raise ValueError("\"" + input_language + "\"", "not in supported languages:", supported_languages)
        elif output_language not in WWUTransphoner.supported_languages:
            raise ValueError("\"" + input_language + "\"", "not in supported languages:", supported_languages)
        else:
            self.target_trie = PhoneTrie(output_language)
            self.input_trie = PhoneTrie(input_language)

            # only can produce sentences for english
            if output_language == 'en':
                self.happy_wp = HappyWordPrediction()
                self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
                self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

            self.input_language = input_language
            self.output_language = output_language


    def set_multipliers(self, imageability=1.0, orthographic=1, phonetic=1.0, semantic=50):
        """
        Update the multipliers that are used when computing how similar a mnemonic is
        to the user's input.

        :param  imageability:
        :param  orthographic:
        :param      phonetic:
        :param      semantic:
        """
        MatchList.aoa_multiplier = imageability
        MatchList.phonetic_multiplier = phonetic
        MatchList.semantic_multiplier = semantic
        MatchList.orthographic_multiplier = orthographic

    def get_mnemonics(self, input_word, translation=None, N=5):
        """
        Return a list of mnemonics similar to the input word

        :param   input_word: the input word for which to return mnemonics
        :param  translation: translation for the input word (optional), (default None)
        :param            N: number of mnemonics to return (default 5)
        :returns           : a list of N mnemonic phrases/words
        :raises    KeyError: raises when the input word's phones are not in the dictionary
        """

        input_node = self.input_trie.search(input_word.lower())
        if not input_node:
            raise KeyError("Can't find phones for input word:", input_word)

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
                    match.search_failed = True
                    match.is_fully_matched = True
                    match_list.add_match(match)
            working_matches = match_list.remove_and_retrieve_unfinished_matches(N)

        return [ match.matched_words for match in match_list.get_finished_matches(N) ]

    def mark_ignored(self, word):
        """
        Mark a word ignored such that it won't be used in further mnemonics

        :param word: the word to be ignored
        """
        node = self.target_trie.search(word)
        if node:
            node.ignored = True

    def __gen_sentence_ends(self, input_mnemonics):
        """
        Return a list of sentences. The list will have one sentence per mnemonic
        found in input_mnemonics.

        :param input_mnemonics: a list of strings containing mnemonics
        :returns              : a list of sentences starting with the strings found in input_mnemonics
        """

        sentences = [] # finished sentences

        # Encodes the input mnemonics into pytorch tensors
        encodings = []
        for input in input_mnemonics:
            encodings.append(self.tokenizer.encode(input, return_tensors='pt')[0])

        # Input mnemonics get encoded into different token lengths, and the model
        # needs equal length inputs for all inputs in a batch so we seperate them out
        # into batches of equal token length to speed up the generation
        token_lenghts = { len(encoding) for encoding in encodings }
        for lenght in token_lenghts:
            curr_list = [encoding for encoding in encodings if len(encoding) == lenght]
            outputs = self.model.generate(torch.stack(curr_list), max_length=15, do_sample=True)
            for tensor in outputs:
                decoded_text = self.tokenizer.decode(tensor, skip_special_tokens=True)
                sentences.append(WWUTransphoner.trim_to_one_sentence(decoded_text))

        return sentences

    def trim_to_one_sentence(text):
        """
        Return the text shortened to one sentence, sentence breaks are denoted by
        a period, exclamation mark, or question mark

        :param text: text to be shortened
        :returns   : text shortened to one sentence
        """

        punctuation = ['.', '!', '?']

        sentence_end = len(text)
        for p in punctuation:
            if p in text:
                sentence_end = min(sentence_end, text.index(p))
        return text[:sentence_end+1]


    def __gen_sentence_beginning(self, incomplete_sentence):
        """
        Return a complete sentence, composed of newly generated text preceeding the
        passed in incomplete sentence

        :param incomplete_sentence: an incomplete sentence that still needs a beginning
        :returns                  : a fully complete sentence
        """

        last_token = ''
        for _ in range(random.randrange(0,8)):
            temp = '[MASK] ' + incomplete_sentence
            result = self.happy_wp.predict_mask(temp, top_k=5) # makes 5 predictions
            new_token = result[random.randrange(0, 5)].token

            # choose another prediction if predictions are repeating
            # (prevents abnormal beginning of sentences)
            attemps = 0
            while (new_token == '•' or last_token == new_token) and attemps < 5:
                new_token = result[random.randrange(0, 5)].token
                last_token = new_token
                attemps += 1

            incomplete_sentence = new_token + ' ' + incomplete_sentence

        return incomplete_sentence

    def gen_sentences(self, input_mnemonics):
        """
        Return a list of mnemonic sentences, one sentence per mnemonic in input_mnemonics

        :param input_mnemonics: a list of mnemonics (strings)
        :raises      TypeError: raises when output language not english, other languages not yet supported
        :returns              : a list of sentences built around the mnemonics in input_mnemonics
        """
        if self.output_language != 'en':
            raise TypeError("Can only generate sentences for output language: 'en'")

        sentence_ends = self.__gen_sentence_ends(input_mnemonics)

        complete_sentences = []
        for incomplete_sentence in sentence_ends:
            complete_sentences.append(self.__gen_sentence_beginning(incomplete_sentence))

        return complete_sentences

a = WWUTransphoner("de", "en")
b = a.get_mnemonics("tropisch", N=10)
print(b)
c = a.gen_sentences(b)
print(c)