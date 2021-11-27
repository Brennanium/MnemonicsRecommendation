from models.autofill import *
from models.ngram import *
from nltk.corpus import arcosg
import time

class Results:
    def __init__(self) -> None:
        self.key_ratio_sum = 0.0
        self.key_ratio_count = 0
        self.word_ratio_sum = 0.0
        self.word_ratio_count = 0
        self.times = []
        self.time_total = 0.0

    def add_key_ratio(self,key_ratio):
        self.key_ratio_sum += key_ratio
        self.key_ratio_count += 1

    def add_time(self,time):
        self.times.append(time)
        self.time_total += time

    def avg_time(self):
        return sum(self.times) / len(self.times)

    def get_key_ratio(self,sentence_total=None):
        self.key_ratio = self.key_ratio_sum / self.key_ratio_count if sentence_total is None else sentence_total
        return self.key_ratio

    def add_word_ratio(self,word_ratio):
        self.word_ratio_sum += word_ratio
        self.word_ratio_count += 1

    def get_word_ratio(self,sentence_total=None):
        self.word_ratio = self.word_ratio_sum / self.word_ratio_count if sentence_total is None else sentence_total
        return self.word_ratio



def evaluate(model: SuggestionModel, sentence, verbose=False): 
    if isinstance(sentence,str):
        words = sentence.split()
    else:
        words = [w for w in sentence if w not in ['<s>','</s>']]

    word_count = len(words)
    key_presses = sum(len(w) for w in words)
    autofill_key_presses = 0
    autofill_count = 0

    if verbose: print([w for w in words])

    for i, w in enumerate(words):
        for j in range(len(w)+1):
            autofill_key_presses += 1

            context = words[0:i]

            suggestions = model.suggestions_context(w[:j],context)

            if verbose: 
                print(w[:j],end='\t')
                print(suggestions)

            if w in suggestions:
                autofill_count += 1
                if verbose: print('found word', w,'!')
                break
        autofill_key_presses -= 1
    
    if verbose: 
        print('key presses no autofill:',key_presses)
        print('key presses with autofill:', autofill_key_presses)
        print('words no autofill:',word_count)
        print('successful autofills:', autofill_count)

    return key_presses, autofill_key_presses, word_count, autofill_count


def split_data(test_files=['c01.txt','f01.txt','n01.txt','p01.txt']):
    test_sents = [s for f in test_files for s in arcosg.sents(f)]

    training_sents = [s for s in arcosg.sents() if s not in test_sents]
    
    return training_sents, test_sents


def test_model(model: SuggestionModel,training_sents=None, test_sents=None,verbose=True):
    if training_sents is None or test_sents is None:
        training_sents, test_sents = split_data()
    
    sent_count = len(test_sents)

    r = Results()
    for i, sent in enumerate(test_sents):
        if verbose: print('\r', i+1,'/',sent_count,sep='',end='')
        
        t = time.process_time()
        out = evaluate(model,sent)
        r.add_time(time.process_time() - t)
        
        r.add_key_ratio(out[1] / out[0])
        r.add_word_ratio(out[3] / out[2])
    print()

    return r

def test_models(models: list[SuggestionModel],training_sents=None, test_sents=None,verbose=True):
    if training_sents is None or test_sents is None:
        training_sents, test_sents = split_data()
    
    sent_count = len(test_sents)

    r = dict((m, Results()) for m in models)
    for i, sent in enumerate(test_sents):
        for j, m in enumerate(models):
            if verbose: print('\r', i+1,'/',sent_count,' model: ',j,sep='',end='')
            
            t = time.process_time()
            out = evaluate(m,sent)
            r[m].add_time(time.process_time() - t)
            
            r[m].add_key_ratio(out[1] / out[0])
            r[m].add_word_ratio(out[3] / out[2])
    print()
    return r
