#!/usr/bin/python

"""trie module example code."""

__author__ = 'Michal Nazarewicz <mina86@mina86.com>'
__copyright__ = 'Copyright 2014 Google Inc.'

# pylint: disable=invalid-name,superfluous-parens

import os
import stat
import sys

import pygtrie


if not os.isatty(0):
    sys.exit(0)


try:
    import termios
    import tty

    def getch():
        """Reads single character from standard input."""
        attr = termios.tcgetattr(0)
        try:
            tty.setraw(0)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(0, termios.TCSADRAIN, attr)

except ImportError:
    try:
        from msvcrt import getch  # pylint: disable=import-error
    except ImportError:
        sys.exit(0)



print('\nDictionary test')
print('===============\n')

t = pygtrie.CharTrie()
t['cat'] = True
t['caterpillar'] = True
t['car'] = True
t['bar'] = True
t['exit'] = False

print('Start typing a word, "exit" to stop')
print('(Other words you might want to try: %s)\n' % ', '.join(sorted(
    k for k in t if k != 'exit')))

text = ''
while True:
    ch = getch()
    if ord(ch) < 32:
        print('Exiting')
        break

    text += ch
    value = t.get(text)
    if value is False:
        print('Exiting')
        break
    if value is not None:
        print(repr(text), 'is a word')
    if t.has_subtrie(text):
        print(repr(text), 'is a prefix of a word')
    else:
        print(repr(text), 'is not a prefix, going back to empty string')
        text = ''