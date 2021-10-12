import nltk
from collections import Counter, defaultdict
import random
import heapq
import operator
import math



class NgramModel:
    def __init__(self, n, dict_model=None, ngrams=None, sents=None):
        if n == 1:
            self.__class__ = UnigramModel
        
        self.n = n
        
        if dict_model is not None:
            self.model = dict_model
            return

        if ngrams is not None:
            self.train_ngrams(n, ngrams)
        elif sents is not None:
            self.train_sents(n, sents)
        else:
            self.model = None

        self.context = []
        self.words = []
        self.words = self.next_words(self.context)


    def train_sents(self, n, sents):
        if not self.n == n:
            return None

        ngrams = [ng for sent in sents for ng in nltk.ngrams(['<s>' for _ in range(n-1)] + sent + ['</s>'],n)]

        return self.train_ngrams(n, ngrams)

    def train_ngrams(self, n, ngrams) -> defaultdict:
        if not self.n == n:
            return None

        self.model = defaultdict(lambda: defaultdict(lambda: 0))

        for ng in ngrams:
            self.model[ng[:-1]][ng[-1]] += 1  # [(w1, w2, ... , wn-1)][wn]
        
        total_count = 0      
        for hist in self.model: # hist = (w1, w2, ... , wn-1)
            total_count = float(sum(self.model[hist].values()))
            for w in self.model[hist]: # w = wn
                self.model[hist][w] /= total_count
    
        return self.model

    def next_words(self,context) -> list[tuple[str,float]]:
        if (not self.context == context) or (self.words is None):
            self.context = context
            self.words = [w for w in self.get_ngram(context).items() if not w[0] in ('<s>','</s>','<UNK>')]

        return self.words

    def next_nwords(self,n,context) -> list[str]:
        top = heapq.nlargest(n, self.next_words(context), key=operator.itemgetter(1))
        return [w[0] for w in top]

    def get_ngram(self, context) -> defaultdict:
        while len(context) < self.n - 1:
            context = ['<s>'] + context
        
        given = tuple(w for w in context[-(self.n-1):])

        return self.model[given]

    def P(self, word, context) -> float:
        if (not self.context == context) or (self.words is None):
            self.context = context
            self.words = self.next_words(self.context)
        
        return next((w[1] for w in self.words if w[0] == word),0)


class UnigramModel(NgramModel):
    def __init__(self, dict_model=None, ngrams=None, sents=None):
        super().__init__(1,dict_model=dict_model,ngrams=ngrams,sents=sents)

    def train_sents(self, n, sents):
        unigrams = [ng for sent in sents for ng in sent]

        return self.train_ngrams(n,unigrams)

    def train_ngrams(self, n, ngrams):
        self.model = defaultdict(lambda: 0)

        self.unigrams_count = len(ngrams)
        self.unigram_fdist = nltk.FreqDist(ngrams)

        for ug, freq in self.unigram_fdist.items(): 
            self.model[ug] = freq/self.unigrams_count
        return self.model

    def get_ngram(self, context) -> defaultdict[str,int]:
        return self.model

    def P(self, word, context) -> float:
        return self.model[word]


class BigramModel(NgramModel):
    def __init__(self, dict_model=None, ngrams=None, sents=None):
        super().__init__(2,dict_model=dict_model,ngrams=ngrams,sents=sents)

class TrigramModel(NgramModel):
    def __init__(self, dict_model=None, ngrams=None, sents=None):
        super().__init__(3,dict_model=dict_model,ngrams=ngrams,sents=sents)


class LinearInterpolationModel(NgramModel):
    def __init__(self, models: list[NgramModel] = None, weights: dict[int,float] = None, ngrams=None, sents=None):
        if models is None:
            models = [
                UnigramModel(ngrams=ngrams,sents=sents),
                BigramModel(ngrams=ngrams,sents=sents),
                TrigramModel(ngrams=ngrams,sents=sents)]
        self.models = models
        self.n_max = max(m.n for m in self.models)

        if weights is None:
            weights = {
                1:0.1,
                2:0.4,
                3:0.5
                }
        self.weights = weights

        if ngrams is not None:
            for m in self.models:
                n = len(ngrams[0])
                if m.n == n: 
                    m.train_ngrams(n,ngrams)
        elif sents is not None:
            for m in self.models: 
                m.train_sents(m.n, sents)
        else:
            self.model = None

        self.context = []
        self.words = []
        self.words = self.next_words(self.context)

    def train_sents(self, n, sents):
        for m in self.models: 
            m.train_sents(m.n, sents)

        return self.models

    def train_ngrams(self, n, ngrams):
        if not n in (m.n for m in self.models):
            return None
        
        for m in self.models:
            n = len(ngrams[0])
            if m.n == n: 
                m.train_ngrams(n,ngrams)
    
        return self.models

    def get_ngram(self, context) -> dict[str,float]:
        while len(context) < self.n_max - 1:
            context = ['<s>'] + context
        
        words = set(w[0] for m in self.models for w in m.next_words(context))
        
        ngrams = [m.n for m in self.models]
        ngrams.sort(reverse=True)

        next_words = dict((m.n, m.get_ngram(context)) for m in self.models)
        probs = {}
        for w in words:
            prob = 0
            for n in ngrams:
                prob += self.weights[n] * next_words[n][w]
            probs[w] = prob

        return probs

class BackoffInterpolationModel(LinearInterpolationModel):
    def get_ngram(self, context) -> dict[str,float]:
        while len(context) < self.n_max - 1:
            context = ['<s>'] + context
        
        words = set(w[0] for m in self.models for w in m.next_words(context))
        
        ngrams = [m.n for m in self.models]
        ngrams.sort(reverse=True)

        next_words = dict((m.n, m.get_ngram(context)) for m in self.models)
        probs = {}
        for w in words:
            for n in ngrams:
                prob = next_words[n][w]
                if not prob == 0:
                    break
            probs[w] = prob

        return probs