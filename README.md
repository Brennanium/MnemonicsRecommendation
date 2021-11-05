# MnemonicsRecommendation

[Google Colab](https://colab.research.google.com/drive/1Pb4GqYC7WWOjLtyeCQ9twlyTK2CTNekw?authuser=1#scrollTo=-ShIETSs4flQ)

Currently the word to find mnemonics for is in the 'Search.py'
file, running that file will print a list of mnemonic matches
sorted by distance from the input word. To run the test's ive
been using: python3 -m unittest TestMatchList.py TestDictionaryTrie.py

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