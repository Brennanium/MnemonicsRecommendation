import nltk
from collections import Counter, defaultdict
import heapq
from operator import itemgetter, attrgetter, methodcaller
from nltk import probability
from nltk.corpus import arcosg
from lexpy._base.automata import FSA
from lexpy.trie import Trie
from lexpy.dawg import DAWG
from models.ngram import *


class SuggestionModel:
    def nsuggestions(self, n, prefix) -> list[str]:
        pass
    
    def suggestions(self, prefix) -> list[str]:
        return self.nsuggestions(3,prefix)

    def suggestions_context(self, prefix, context) -> list[str]:
        return self.nsuggestions(3,prefix)

    def nsuggestions_context(self, n, prefix, context) -> list[str]:
        return self.nsuggestions(n,prefix)



class FSAModel(SuggestionModel):
    def __init__(self, words: list[str] = None, fsa: FSA = None) -> None:
        if fsa is None:
            self.fsa = DAWG()
        else:
            self.fsa = fsa

        self.add_all(words)
        
    def add_all(self,words=None):
        if words is None:
            self.fsa.add_all([w for w in arcosg.words()])
        else:
            self.fsa.add_all(words)

    def P(self, count): 
        return count / self.fsa.get_word_count()

    def nsuggestions(self, n, prefix) -> list[str]:
        if prefix == '':
            suggestions = self.fsa.search('*',with_count=True)
        else: 
            suggestions = self.fsa.search_with_prefix(prefix,with_count=True)
        top = heapq.nlargest(n,suggestions,key=lambda x: self.P(x[1]))
        return [s[0] for s in top]


class TrieModel(FSAModel):
    def __init__(self, words: list[str] = None) -> None:
        super().__init__(words=words,fsa=Trie())
       
class MinimalDFAModel(FSAModel):
    def __init__(self, words: list[str] = None) -> None:
        super().__init__(words=words,fsa=DAWG())
       


class NgramFSAModel(FSAModel):
    def __init__(self, words: list[str] = None, ngram: NgramModel = None, fsa: FSA = None) -> None:
        if ngram:
            self.ngram = ngram
        
        super().__init__(words=words,fsa=fsa)
        
    def set_model(self, ngram: NgramModel):
        self.ngram = ngram

    def suggestions_context(self, prefix, context) -> list[str]:
        return self.nsuggestions_context(3,prefix,context)

    def nsuggestions_context(self, n, prefix, context) -> list[str]:
        if prefix == '':
            return self.ngram.next_nwords(n,context)
        else: 
            suggestions = self.fsa.search_with_prefix(prefix,with_count=True)

        probabilities = [(s[0], self.ngram.P(s[0],context)) for s in suggestions]

        top = heapq.nlargest(n,probabilities,key=lambda x: x[1])

        return [s[0] for s in top if not s[1] == 0]

class NgramTrieModel(NgramFSAModel):
    def __init__(self, words: list[str] = None, ngram: NgramModel = None) -> None:
        super().__init__(words=words,ngram=ngram,fsa=Trie())

class NgramMinimalDFAModel(NgramFSAModel):
    def __init__(self, words: list[str] = None, ngram: NgramModel = None) -> None:
        super().__init__(words=words,ngram=ngram,fsa=DAWG())