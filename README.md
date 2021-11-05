# MnemonicsRecommendation

Main feature of this repo is the WWUTransphoner class found in the Search file.
There are 5 languages this product is supposed to support {'de', 'zh', 'fr', 'ja', 'en'}
Currently we can only produce sentences useful for english speakers, meaning our system can produce mnemonic phrases/sentences for learning words in German, Japanese and English.

Example usage:
```python
wwut = WWUTransphoner('de") # can create mnemonics for german words

# Must supply the word to build a mnemonic for, and it's translation
mnemonics = wwut.get_mnemonics('tropisch', 'tropical') 

sentences = SentenceGen.gen_sentences(mnemonics)
```

Mnemonics will contain
* 'trip ash'
* 'troop ash'
* 'true push'
* 'trap ash'
* 'true piece'

And sentences will contain
* 'trip ash at her feet, but instead of answering, he turned the pages'
* 'nothing but mom and i troop ash, but you can stay up here'
* 'frantic and desperately desperate to find his true push, and i 'd do anything for him'
* 'beneath his hands and my two fingers trap ash's fingers'


Word2Vec embeddings taken from:
https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

Phonetic Lists taken from:
https://github.com/open-dict-data/ipa-dict#csv

English age of aquisition ratings taken from Kuperman et all

Aline algorithm and feature comparison method taken from:
https://www.nltk.org/_modules/nltk/metrics/aline.html
We modified it to have more phones that it can compare, and can
handle extra cases where different dictionaries use different
character encodings for the same phone
