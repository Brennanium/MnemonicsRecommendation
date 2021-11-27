# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nltk
from collections import Counter, defaultdict
import heapq
from operator import itemgetter, attrgetter, methodcaller
from nltk.corpus import arcosg
import regex as re

# %% [markdown]
# # Norvig's Spelling Corrector

# %%
def words(text): return re.findall(r'\w+', text.lower())

# WORDS = Counter(words(open('big.txt').read()))
WORDS = Counter(arcosg.words())

def P(word, N=sum(WORDS.values())): 
    # Probability of `word`.
    return WORDS[word] / N

def correction(word): 
    # Most probable spelling correction for word.
    return heapq.nlargest(5,candidates(word), key=P)

def candidates(word): 
    # Generate possible spelling corrections for word.
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    # The subset of `words` that appear in the dictionary of WORDS.
    return set(w for w in words if w in WORDS)

def edits1(word):
    # All edits that are one edit away from `word`.
    letters    = "aàbcdeèfghiìjklmnoòpqrstuùvwxyz'-_"
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    # All edits that are two edits away from `word`.
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# %%
correction('anis')


# %%
correction('ab')

# %% [markdown]
# ## Resources 
# http://norvig.com/spell-correct.html
# %% [markdown]
# # LinSpell

# %%
# /// <summary>
# /// Computes and returns the Damerau-Levenshtein edit distance between two strings, 
# /// i.e. the number of insertion, deletion, sustitution, and transposition edits
# /// required to transform one string to the other. This value will be >= 0, where 0
# /// indicates identical strings. Comparisons are case sensitive, so for example, 
# /// "Fred" and "fred" will have a distance of 1. This algorithm is basically the
# /// Levenshtein algorithm with a modification that considers transposition of two
# /// adjacent characters as a single edit.
# /// http://blog.softwx.net/2015/01/optimizing-damerau-levenshtein_15.html
# /// https://github.com/softwx/SoftWx.Match
# /// </summary>
# /// <remarks>See http://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
# /// This is inspired by Sten Hjelmqvist'string1 "Fast, memory efficient" algorithm, described
# /// at http://www.codeproject.com/Articles/13525/Fast-memory-efficient-Levenshtein-algorithm.
# /// This version differs by adding additiona optimizations, and extending it to the Damerau-
# /// Levenshtein algorithm.
# /// Note that this is the simpler and faster optimal string alignment (aka restricted edit) distance
# /// that difers slightly from the classic Damerau-Levenshtein algorithm by imposing the restriction
# /// that no substring is edited more than once. So for example, "CA" to "ABC" has an edit distance
# /// of 2 by a complete application of Damerau-Levenshtein, but a distance of 3 by this method that
# /// uses the optimal string alignment algorithm. See wikipedia article for more detail on this
# /// distinction.
# /// </remarks>
# /// <license>
# /// The MIT License (MIT)
# ///
# ///Copyright(c) 2015 Steve Hatchett
# ///
# ///Permission is hereby granted, free of charge, to any person obtaining a copy
# ///of this software and associated documentation files(the "Software"), to deal
# ///in the Software without restriction, including without limitation the rights
# ///to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# ///copies of the Software, and to permit persons to whom the Software is
# ///furnished to do so, subject to the following conditions:
# ///
# ///The above copyright notice and this permission notice shall be included in all
# ///copies or substantial portions of the Software.
# ///
# ///THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# ///IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# ///FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# ///AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# ///LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# ///OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# ///SOFTWARE.
# /// </license>


# /// <param name="string1">String being compared for distance.</param>
# /// <param name="string2">String being compared against other string.</param>
# /// <param name="maxDistance">The maximum edit distance of interest.</param>
# /// <returns>int edit distance, >= 0 representing the number of edits required
# /// to transform one string to the other, or -1 if the distance is greater than the specified maxDistance.</returns>
def DamerauLevenshteinDistance(string1, string2, maxDistance):
    if string1 is None or len(string1) == 0: return 0 if string2 is None else len(string2)
    if string2 is None or len(string2) == 0: return len(string1)

    # if strings of different lengths, ensure shorter string is in string1. This can result in a little
    # faster speed by spending more time spinning just the inner loop during the main processing.
    # swap string1 and string2
    if len(string1) > len(string2):
        temp = string1
        string1 = string2
        string2 = temp 

    sLen = len(string1) # this is also the minimun length of the two strings
    tLen = len(string2)

    #  suffix common to both strings can be ignored
    while ((sLen > 0) and (string1[sLen - 1] == string2[tLen - 1])): 
        sLen -= 1 
        tLen -= 1

    start = 0
    if ((string1[0] == string2[0]) or (sLen == 0)): # if there'string1 a shared prefix, or all string1 matches string2'string1 suffix
                                                    # prefix common to both strings can be ignored
        while ((start < sLen) and (string1[start] == string2[start])): start += 1

        sLen -= start # length of the part excluding common prefix and suffix
        tLen -= start

        # if all of shorter string matches prefix and/or suffix of longer string, then
        # edit distance is just the delete of additional characters present in longer string
        if (sLen == 0): return tLen

        string2 = string2[start:(start+tLen)] # faster than string2[start+j] in inner loop below
    
    lenDiff = tLen - sLen
    if ((maxDistance < 0) or (maxDistance > tLen)):
        maxDistance = tLen
    elif (lenDiff > maxDistance): return -1

    v0 = [0 for i in range(tLen)]
    v2 = [0 for i in range(tLen)]  # stores one level further back (offset by +1 position)
    j = 0
    while j < maxDistance: 
        v0[j] = j + 1
        # v0.append(j + 1)
        j += 1
    while j < tLen: 
        v0[j] = maxDistance + 1
        # v0.append(maxDistance + 1)
        j += 1
    
    jStartOffset = maxDistance - (tLen - sLen)
    haveMax = maxDistance < tLen
    jStart = 0
    jEnd = maxDistance
    sChar = string1[0]
    current = 0
    for i in range(0, sLen):
        prevsChar = sChar
        sChar = string1[start + i]
        tChar = string2[0]
        left = i
        current = left + 1
        nextTransCost = 0
        # no need to look beyond window of lower right diagonal - maxDistance cells (lower right diag is i - lenDiff)
        # and the upper left diagonal + maxDistance cells (upper left is i)
        jStart += int(i > jStartOffset)
        jEnd += int(jEnd < tLen)
        j = jStart
        while j < jEnd:
            above = current
            thisTransCost = nextTransCost
            nextTransCost = v2[j]
            v2[j] = current = left  # cost of diagonal (substitution)
            left = v0[j]            # left now equals current cost (which will be diagonal at next iteration)
            prevtChar = tChar
            tChar = string2[j]
            if (sChar != tChar):
                if (left < current): current = left     # insertion
                if (above < current): current = above   # deletion
                current += 1
                if ((i != 0) and (j != 0) and (sChar == prevtChar) and (prevsChar == tChar)):
                    thisTransCost += 1
                    if (thisTransCost < current): current = thisTransCost    # transposition
            v0[j] = current
            j += 1
        if (haveMax and (v0[i + lenDiff] > maxDistance)): return -1
    return current if (current <= maxDistance) else -1


# %%
editDistanceMax=2
# verbose = 1
# 0: top suggestion
# 1: all suggestions of smallest edit distance   
# 2: all suggestions <= editDistanceMax (slower, no early termination)

# public class SuggestItem
# {
#     public string term = "";
#     public int distance = 0;
#     public Int64 count = 0;

#     public override bool Equals(object obj)
#     {
#         return Equals(term, ((SuggestItem)obj).term);
#     }
    
#     public override int GetHashCode()
#     {
#         return term.GetHashCode();
#     }
# }



unigrams = [ug for sent in arcosg.sents() for ug in sent]
# unigram_fdist = nltk.FreqDist(unigrams)

# dictionaryLinear = {}
dictionaryLinear = nltk.FreqDist(unigrams)
# dictionaryLinear = WORDS
maxlength = 0; # maximum dictionary term length


def LookupLinear(input, editDistanceMax, verbose=0):
    suggestions = []

    editDistanceMax2 = editDistanceMax

    if verbose < 2 and (count := dictionaryLinear[input]) > 0:
        return (input, count, 0)

    for key, value in dictionaryLinear.items():
        if abs(len(key) - len(input)) > editDistanceMax2: continue

        # if already ed1 suggestion, there can be no better suggestion with smaller count: no need to calculate damlev
        if ((verbose == 0) and (len(suggestions) > 0) and (suggestions[0][2] == 1) and (value <= suggestions[0][1])):  continue

        distance = DamerauLevenshteinDistance(input, key, editDistanceMax2); 
        # sometimes DamerauLevenshteinDistance returnes a distance > editDistanceMax
        if ((distance >= 0) and (distance <= editDistanceMax)):
            # v0: clear if better ed or better ed+count; 
            # v1: clear if better ed                    
            # v2: all

            # do not process higher distances than those already found, if verbose<2
            if ((verbose < 2) and (len(suggestions) > 0) and (distance > suggestions[0][2])): continue

            # we will calculate DamLev distance only to the smallest found distance sof far
            if (verbose < 2): editDistanceMax2 = distance

            # remove all existing suggestions of higher distance, if verbose<2
            if ((verbose < 2) and (len(suggestions) > 0) and (suggestions[0][2] > distance)): suggestions = []

            suggestions.append((key, value, distance))
    

    if (verbose < 2): 
        # sort by descending word frequency
        suggestions.sort(key=lambda x: -x[1]) 
    else: 
        # sort by ascending edit distance, then by descending word frequency
        # suggestions.Sort((x, y) => 2 * x.distance.CompareTo(y.distance) - x.count.CompareTo(y.count));
        suggestions.sort(key=lambda x: (x[2], -x[1]))
    
    if ((verbose == 0) and (len(suggestions) > 1)):
        return suggestions[0:1]
    else: 
        return suggestions


# %%
LookupLinear("anis",2,verbose=1)


# %%
## Resources
# https://github.com/wolfgarbe/LinSpell/blob/master/linspell/LinSpell.cs


