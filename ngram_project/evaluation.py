# test how many keystrokes are saved; 
# if correct word is in top 3 suggestions

# %%
from models.autofill import *
from models.ngram import *
from models.evaluate import *
from nltk.corpus import arcosg

# # %%
# evaluate(TrieModel(), "tha mi a' goid")
# # %%
# evaluate(TrieModel(), "a bheil thu a' faireachdainn math")
# # %%
# evaluate(TrieModel(), arcosg.sents()[0])
# # %%
# evaluate(TrieModel(), load_file()[1])
# # %%
# bigram = BigramModel(sents=arcosg.sents())
# # %%
# ngram_trie = NgramTrieModel(ngram=bigram)
# # %%
# fourgram = NgramModel(4,sents=arcosg.sents())
# # %%
# fourgram.next_nwords(3,['mi',"a'",'creidsinn'])
# # %%
# fourgram.next_nwords(3,['tha'])

# %%
# lin_interp = BackoffInterpolationModel(sents=arcosg.sents())
# print(lin_interp.next_nwords(6,['mi',"a'",'creidsinn']))
# print(lin_interp.next_nwords(10,['tha']))
# %%
evaluate(TrieModel(words=list(arcosg.words())),
    "Tha mi a' faicinn a' ghrian blàth")
# %%
evaluate(NgramTrieModel(words=list(arcosg.words()),ngram=BigramModel(sents=arcosg.sents())),
    "Tha mi a' faicinn a' ghrian blàth")
# %%
evaluate(NgramTrieModel(words=list(arcosg.words()),ngram=LinearInterpolationModel(sents=arcosg.sents())),
    "Tha mi a' faicinn a' ghrian blàth")
# %%
