
"""
Trigram counting
Usage: python count_trigrams.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys

import regex


def tokenize(text):
    words = regex.findall("\p{L}+", text)
    return words


def count_trigrams(words):
    trigrams = [tuple(words[inx:inx + 3])
                for inx in range(len(words) - 2)]
    frequencies = {}
    for trigram in trigrams:
        if trigram in frequencies:
            frequencies[trigram] += 1
        else:
            frequencies[trigram] = 1
    return frequencies


if __name__ == '__main__':
    text1 = """Hejsan mitt namn är "Gustav", vad heter du? Hej! Jag är 34 år gammal och är en "väldigt" glad 
        människa! Är du också - glad? Det hoppas jag. Hejsan mitt namn är
        - Men käre far."""
    text2 = open('Selma.txt').read().lower()
    words = tokenize(text2.lower())
    frequency_trigrams = count_trigrams(words)
    for trigram in frequency_trigrams:
        print(frequency_trigrams[trigram], "\t", trigram)