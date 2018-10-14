
"""
Bigram counting
Usage: python count_bigrams.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys

import regex


def tokenize(text):
    words = regex.findall("\p{L}+", text)
    print(len(words))
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


if __name__ == '__main__':
    text1 = """Hejsan mitt namn är "Gustav", vad heter du? Hej! Jag är 34 år gammal och är en "väldigt" glad 
    människa! Är du också - glad? Det hoppas jag.
    - Men käre far."""
    text2 = open('Selma.txt').read().lower()
    words = tokenize(text1)
    frequency_bigrams = count_bigrams(words)
    for bigram in frequency_bigrams:
        print(frequency_bigrams[bigram], "\t", bigram)

    print(len(bigram))