from PhoneTrie import PhoneTrie
import os
import csv
import sqlite3 as sl

file = open('dictionaries/german/phones_de.csv')
newfile = open('dictionaries/german/phones_de_new.csv', 'w')
for line in file:
    pos = line.find(',')
    newline = line[:pos]
    newline = newline + ';' + line[pos+1:-1].replace('\"', '').replace(' ', '') + ';0\n'
    newfile.write(newline)
newfile.close()