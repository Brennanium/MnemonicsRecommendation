# Origonal algorithm by Grzegorz Kondrak, modified by the
# WWU Mnemonics Recommendation Team to support additional
# phones when computing the delta, and aligning.
# Changes in the code are noted above the additions.
#
#
#
# Natural Language Toolkit: ALINE
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Greg Kondrak <gkondrak@ualberta.ca>
#         Geoff Bacon <bacon@berkeley.edu> (Python port)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
ALINE
https://webdocs.cs.ualberta.ca/~kondrak/
Copyright 2002 by Grzegorz Kondrak.

ALINE is an algorithm for aligning phonetic sequences, described in [1].
This module is a port of Kondrak's (2002) ALINE. It provides functions for
phonetic sequence alignment and similarity analysis. These are useful in
historical linguistics, sociolinguistics and synchronic phonology.

ALINE has parameters that can be tuned for desired output. These parameters are:
- C_skip, C_sub, C_exp, C_vwl
- Salience weights
- Segmental features

In this implementation, some parameters have been changed from their default
values as described in [1], in order to replicate published results. All changes
are noted in comments.

Example usage
-------------

# Get optimal alignment of two phonetic sequences

>>> align('θin', 'tenwis') # doctest: +SKIP
[[('θ', 't'), ('i', 'e'), ('n', 'n'), ('-', 'w'), ('-', 'i'), ('-', 's')]]

[1] G. Kondrak. Algorithms for Language Reconstruction. PhD dissertation,
University of Toronto.
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy import ndarray

try:
    import numpy as np
    from numpy import ndarray
except ImportError:
    np = None

from typing import List, Union, Tuple, Dict, Optional

# === Constants ===

inf = float("inf")

# Default values for maximum similarity scores (Kondrak 2002: 54)
C_skip = 10  # Indels
C_sub = 35  # Substitutions
C_exp = 45  # Expansions/compressions
C_vwl = 5  # Vowel/consonant relative weight (decreased from 10)

consonants: List[str] = [
    "B",
    "N",
    "R",
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "x",
    "z",
    "ç",
    "ð",
    "ħ",
    "ŋ",
    "ɖ",
    "ɟ",
    "ɢ",
    "ɣ",
    "ɦ",
    "ɬ",
    "ɮ",
    "ɰ",
    "ɱ",
    "ɲ",
    "ɳ",
    "ɴ",
    "ɸ",
    "ɹ",
    "ɻ",
    "ɽ",
    "ɾ",
    "ʀ",
    "ʁ",
    "ʂ",
    "ʃ",
    "ʈ",
    "ʋ",
    "ʐ ",
    "ʒ",
    "ʔ",
    "ʕ",
    "ʙ",
    "ʝ",
    "β",
    "θ",
    "χ",
    "ʐ",
    "w",
    # below are the additional phones added, not from origonial algorithm
    "ɡ",
    "ɕ",
    "ɵ",
    "ʧ",
    "ɭ",
    "ʑ",
    "Q"
]

# Relevant features for comparing consonants and vowels
R_c: List[str] = [
    "aspirated",
    "lateral",
    "manner",
    "nasal",
    "place",
    "retroflex",
    "syllabic",
    "voice",
    "rhotic",   # added seperate rhotic feature
]
# 'high' taken out of R_v because same as manner
R_v: List[str] = [
    "back",
    "lateral",
    "long",
    "manner",
    "nasal",
    "place",
    "retroflex",
    "round",
    "syllabic",
    "voice",
]

# Flattened feature matrix (Kondrak 2002: 56)
similarity_matrix: Dict[str,float] = {
    # place
    "bilabial": 1.0,
    "labiodental": 0.95,
    "dental": 0.9,
    "alveolar": 0.85,
    "retroflex": 0.8,
    "palato-alveolar": 0.75,
    "palatal": 0.7,
    "velar": 0.6,
    "uvular": 0.5,
    "pharyngeal": 0.3,
    "glottal": 0.1,
    "labiovelar": 1.0,
    "vowel": -1.0,  # added 'vowel'
    # manner
    "stop": 1.0,
    "affricate": 0.9,
    "fricative": 0.85,  # increased fricative from 0.8
    "trill": 0.7,
    "tap": 0.65,
    "approximant": 0.6,
    "high vowel": 0.4,
    "mid vowel": 0.2,
    "low vowel": 0.0,
    "vowel2": 0.5,  # added vowel
    # high
    "high": 1.0,
    "mid": 0.5,
    "low": 0.0,
    # back
    "front": 1.0,
    "central": 0.5,
    "back": 0.0,
    # binary features
    "plus": 1.0,
    "minus": 0.0,
}

# Relative weights of phonetic features (Kondrak 2002: 55)
salience: Dict[str,int] = {
    "syllabic": 15, # increased from 5
    "place": 60,  # increased from 40
    "manner": 40, # decreased from 50
    "voice": 10,  # decreased from 10 # increased back to 10 from 5
    "nasal": 20,  # increased from 10
    "retroflex": 10,
    "lateral": 10,
    "aspirated": 5,
    "long": 0,  # decreased from 1
    "high": 3,  # decreased from 5
    "back": 2,  # decreased from 5
    "round": 10,  # decreased from 5 # increased to 10 from 2
    "rhotic": 30,
}

# (Kondrak 2002: 59-60)
feature_matrix: Dict[str,Dict[str,str]] = {
    # Consonants
    "p": {
        "place": "bilabial",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "b": {
        "place": "bilabial",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "t": {
        "place": "alveolar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "d": {
        "place": "alveolar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʈ": {
        "place": "retroflex",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɖ": {
        "place": "retroflex",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "c": {
        "place": "palatal",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɟ": {
        "place": "palatal",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "k": {
        "place": "velar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "g": {
        "place": "velar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "q": {
        "place": "uvular",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɢ": {
        "place": "uvular",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʔ": {
        "place": "glottal",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "m": {
        "place": "bilabial",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɱ": {
        "place": "labiodental",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "n": {
        "place": "alveolar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɳ": {
        "place": "retroflex",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɲ": {
        "place": "palatal",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ŋ": {
        "place": "velar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɴ": {
        "place": "uvular",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "N": {
        "place": "uvular",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "plus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʙ": {
        "place": "bilabial",
        "manner": "trill",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "B": {
        "place": "bilabial",
        "manner": "trill",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "r": {
        "place": "alveolar",
        "manner": "trill",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ʀ": {
        "place": "uvular",
        "manner": "trill",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "R": {
        "place": "uvular",
        "manner": "trill",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɾ": {
        "place": "alveolar",
        "manner": "tap",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɽ": {
        "place": "retroflex",
        "manner": "tap",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɸ": {
        "place": "bilabial",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "β": {
        "place": "bilabial",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "f": {
        "place": "labiodental",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "v": {
        "place": "labiodental",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "θ": {
        "place": "dental",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ð": {
        "place": "dental",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "s": {
        "place": "alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "z": {
        "place": "alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʃ": {
        "place": "palato-alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʒ": {
        "place": "palato-alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʂ": {
        "place": "retroflex",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʐ": {
        "place": "retroflex",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ç": {
        "place": "palatal",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʝ": {
        "place": "palatal",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "x": {
        "place": "velar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɣ": {
        "place": "velar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "χ": {
        "place": "uvular",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʁ": {
        "place": "uvular",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ħ": {
        "place": "pharyngeal",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʕ": {
        "place": "pharyngeal",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "h": {
        "place": "glottal",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɦ": {
        "place": "glottal",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɬ": {
        "place": "alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɮ": {
        "place": "alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʋ": {
        "place": "labiodental",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɹ": {
        "place": "alveolar",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɻ": {
        "place": "retroflex",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "j": {
        "place": "palatal",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɰ": {
        "place": "velar",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "l": {
        "place": "alveolar",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "w": {
        "place": "labiovelar",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    # Vowels
    "i": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "y": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "e": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "E": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "front",
        "round": "minus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ø": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "front",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɛ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "œ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "front",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "æ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "a": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "A": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "front",
        "round": "minus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɨ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʉ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "central",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ə": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "U": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "back",
        "round": "plus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "o": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "back",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "O": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "back",
        "round": "plus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɔ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "back",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɒ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "back",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "I": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "minus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    # Below are the additional entries not in origonial algorithm
    "ɡ": {
        "place": "velar",
        "manner": "stop",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɭ": {
        "place": "retroflex",
        "manner": "approximant",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ĭ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "minus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʏ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ǝ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɘ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "u": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "back",
        "round": "plus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ŏ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "back",
        "round": "plus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɪ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "minus",
        "long": "plus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɐ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɫ": {
        "place": "alveolar",
        "manner": "approximant",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "back",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʊ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "back",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɑ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "low",
        "back": "back",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɝ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "high": "central",
        "back": "mid",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɚ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "high": "mid",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ʌ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "high": "central",
        "back": "back",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɜ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "high": "central",
        "back": "mid",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ɕ": {
        "place": "palato-alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "high": "high",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ᵻ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "plus",
        "lateral": "minus",
        "high": "high",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "plus",
    },
    "ʧ": {
        "place": "palato-alveolar",
        "manner": "affricate",
        "syllabic": "minus",
        "voice": "minus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʤ": {
        "place": "palato-alveolar",
        "manner": "affricate",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "central",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ɯ": {
        "place": "vowel",
        "manner": "vowel2",
        "syllabic": "plus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "high",
        "back": "back",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    },
    "ʑ": {
        "place": "palato-alveolar",
        "manner": "fricative",
        "syllabic": "minus",
        "voice": "plus",
        "nasal": "minus",
        "retroflex": "minus",
        "lateral": "minus",
        "high": "mid",
        "back": "front",
        "round": "minus",
        "long": "minus",
        "aspirated": "minus",
        "rhotic": "minus",
    }
}

# === Algorithm ===


def align(str1: str, str2: str, epsilon: Optional[float]=0):
    """
    Compute the alignment of two phonetic strings.

    :type str1, str2: str
    :param str1, str2: Two strings to be aligned
    :type epsilon: float (0.0 to 1.0)
    :param epsilon: Adjusts threshold similarity score for near-optimal alignments

    :rtpye: list(list(tuple(str, str)))
    :return: Alignment(s) of str1 and str2

    (Kondrak 2002: 51)
    """
    if np is None:
        raise ImportError("You need numpy in order to use the align function")

    assert 0.0 <= epsilon <= 1.0, "Epsilon must be between 0.0 and 1.0."

    str1 = str1.replace('dʒ','ʤ')
    str2 = str2.replace('dʒ','ʤ')
    str1 = str1.replace('tʃ','ʧ')
    str2 = str2.replace('tʃ','ʧ')

    m = len(str1)
    n = len(str2)
    # This includes Kondrak's initialization of row 0 and column 0 to all 0s.
    S = np.zeros((m + 1, n + 1), dtype=float)

    # If i <= 1 or j <= 1, don't allow expansions as it doesn't make sense,
    # and breaks array and string indices. Make sure they never get chosen
    # by setting them to -inf.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            edit1 = S[i - 1, j] + sigma_skip(str1[i - 1])
            edit2 = S[i, j - 1] + sigma_skip(str2[j - 1])
            edit3 = S[i - 1, j - 1] + sigma_sub(str1[i - 1], str2[j - 1])
            if i > 1:
                edit4 = S[i - 2, j - 1] + sigma_exp(str2[j - 1], str1[i - 2 : i])
            else:
                edit4 = -inf
            if j > 1:
                edit5 = S[i - 1, j - 2] + sigma_exp(str1[i - 1], str2[j - 2 : j])
            else:
                edit5 = -inf
            S[i, j] = max(edit1, edit2, edit3, edit4, edit5, 0)

    T = (1 - epsilon) * np.amax(S)  # Threshold score for near-optimal alignments

    alignments = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S[i, j] >= T:
                alignments.append(_retrieve(i, j, 0, S, T, str1, str2, []))
    return alignments



def _retrieve(i: int, j: int, s: float, S: ndarray, T: float, str1: str, str2: str, out: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    """
    Retrieve the path through the similarity matrix S starting at (i, j).

    :rtype: list(tuple(str, str))
    :return: Alignment of str1 and str2
    """
    if S[i, j] == 0:
        return out
    else:
        if j > 1 and S[i - 1, j - 2] + sigma_exp(str1[i - 1], str2[j - 2 : j]) + s >= T:
            out.insert(0, (str1[i - 1], str2[j - 2 : j]))
            _retrieve(
                i - 1,
                j - 2,
                s + sigma_exp(str1[i - 1], str2[j - 2 : j]),
                S,
                T,
                str1,
                str2,
                out,
            )
        elif (
            i > 1 and S[i - 2, j - 1] + sigma_exp(str2[j - 1], str1[i - 2 : i]) + s >= T
        ):
            out.insert(0, (str1[i - 2 : i], str2[j - 1]))
            _retrieve(
                i - 2,
                j - 1,
                s + sigma_exp(str2[j - 1], str1[i - 2 : i]),
                S,
                T,
                str1,
                str2,
                out,
            )
        elif S[i, j - 1] + sigma_skip(str2[j - 1]) + s >= T:
            out.insert(0, ("-", str2[j - 1]))
            _retrieve(i, j - 1, s + sigma_skip(str2[j - 1]), S, T, str1, str2, out)
        elif S[i - 1, j] + sigma_skip(str1[i - 1]) + s >= T:
            out.insert(0, (str1[i - 1], "-"))
            _retrieve(i - 1, j, s + sigma_skip(str1[i - 1]), S, T, str1, str2, out)
        elif S[i - 1, j - 1] + sigma_sub(str1[i - 1], str2[j - 1]) + s >= T:
            out.insert(0, (str1[i - 1], str2[j - 1]))
            _retrieve(
                i - 1,
                j - 1,
                s + sigma_sub(str1[i - 1], str2[j - 1]),
                S,
                T,
                str1,
                str2,
                out
            )
    return out

def sigma_skip(p: str):
    """
    Returns score of an indel of P.

    (Kondrak 2002: 54)
    """
    return C_skip



def sigma_sub(p: str, q: str):
    """
    Returns score of a substitution of P with Q.

    (Kondrak 2002: 54)
    """
    return C_sub - delta(p, q) - V(p) - V(q)



def sigma_exp(p: str, q: str):
    """
    Returns score of an expansion/compression.

    (Kondrak 2002: 54)
    """
    q1 = q[0]
    q2 = q[1]
    return C_exp - delta(p, q1) - delta(p, q2) - V(p) - max(V(q1), V(q2))



def delta(p: str, q: str):
    """
    Return weighted sum of difference between P and Q.

    (Kondrak 2002: 54)
    """
    features = R(p, q)
    total = 0
    for f in features:
        total += diff(p, q, f) * salience[f]
    return total



def diff(p: str, q: str, f: str):
    """
    Returns difference between phonetic segments P and Q for feature F.

    (Kondrak 2002: 52, 54)
    """
    p_features, q_features = feature_matrix[p], feature_matrix[q]
    return abs(similarity_matrix[p_features[f]] - similarity_matrix[q_features[f]])



def R(p: str, q: str):
    """
    Return relevant features for segment comparsion.

    (Kondrak 2002: 54)
    """
    if p in consonants or q in consonants:
        return R_c
    return R_v



def V(p: str):
    """
    Return vowel weight if P is vowel.

    (Kondrak 2002: 54)
    """
    if p in consonants:
        return 0
    return C_vwl
