# MnemonicsRecommendation

Main feature of this repo is the WWUTransphoner class found in the Search file.
There are 5 languages this product is supposed to support {'de', 'zh', 'fr', 'ja', 'en'}
Currently we can only produce sentences useful for english speakers, meaning our system can produce mnemonic phrases/sentences for learning words in German, Japanese and English.

# Example usage:
***
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

#Website Usage:
***
```
$ cd MnemonicsRecommendation
$ flask run
```


Word2Vec embeddings from:
[wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) ([Apache License](https://www.apache.org/licenses/LICENSE-2.0))
and
[Global Vectors for Word Representation](https://github.com/stanfordnlp/GloVe) ([Apache License](https://www.apache.org/licenses/LICENSE-2.0))

Phonetic Lists from:
https://github.com/open-dict-data/ipa-dict#csv
  French lists built from [gc-ipa](https://github.com/dohliam/qc-ipa) ([MIT License](https://github.com/lingz/cmudict-ipa/blob/master/LICENSE))
  Japanese lists built from [edict](https://www.edrdg.org/jmdict/edict.html) ([CC BY-SA 3.0](https://creativecommons.org/licenses/by/4.0/))
  English lists built from [cmudict-ipa](https://github.com/lingz/cmudict-ipa) ([MIT License](https://github.com/lingz/cmudict-ipa/blob/master/LICENSE))
  German lists built from [germanipa](https://github.com/kdelaney/germanipa)

English age of aquisition ratings from Kuperman et all

Aline algorithm and feature comparison method from:
https://www.nltk.org/_modules/nltk/metrics/aline.html
We modified it to have more phones that it can compare, and can
handle extra cases where different dictionaries use different
character encodings for the same phone


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Citations:
***
Bird, Steven, Edward Loper and Ewan Klein (2009).
Natural Language Processing with Python.  O'Reilly Media Inc.

Kuperman, V., Stadthagen-Gonzalez, H. & Brysbaert, M. Age-of-acquisition ratings for 30,000 English words. Behav Res 44, 978–990 (2012). https://doi.org/10.3758/s13428-012-0210-4

Yamada, Ikuya, et al. "Wikipedia2Vec: An efficient toolkit for learning and visualizing the embeddings of words and entities from Wikipedia." arXiv preprint arXiv:1812.06280 (2018).
