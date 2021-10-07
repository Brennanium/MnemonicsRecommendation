from DictionaryTrie import DictTrie

en_trie = DictTrie('en')

file = open('dictionaries/english/lemmas_no_hyphens.csv')
new_file = open('dictionaries/english/lemmas_no_hyphens_with_ipa.csv', "w+")
for line in file:
    node = en_trie.search(line.strip())
    if node:
        new_file.write(node.word + "\t" + node.phones + "\n")
