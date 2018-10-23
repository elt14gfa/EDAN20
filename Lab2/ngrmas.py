"""
Counting n-grams of any size: N
Usage: python count_ngrams.py N < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys

import regex


def tokenize(text):
    words = regex.findall('\p{L}+', text)
    return words


def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies


def count_ngrams(words, n):
    ngrams = [tuple(words[inx:inx + n])
              for inx in range(len(words) - n + 1)]
    # "\t".join(words[inx:inx + n])
    frequencies = {}
    for ngram in ngrams:
        if ngram in frequencies:
            frequencies[ngram] += 1
        else:
            frequencies[ngram] = 1
    return frequencies


if __name__ == '__main__':

    text1 = """Hejsan mitt namn är "Gustav", vad heter du? Hej! Jag är 34 år gammal och är en "väldigt" glad 
            människa! Är du också - glad? Det hoppas jag. Hejsan mitt namn är
            - Men käre far."""
    text2 = open('Selma.txt').read().lower()
    n = 4
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    words = tokenize(text2)
    # frequency = count_bigrams(words)
    frequency = count_ngrams(words, n)
    for word in sorted(frequency, key=frequency.get, reverse=True):
        print(word, '\t', frequency[word])