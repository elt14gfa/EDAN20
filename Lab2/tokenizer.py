"""
Tokenizers
Usage: python tokenizer.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys
import regex as re
import os

text = open('Selma.txt').read()
text2 = """hejsan mitt namn är Gustav, vad heter du? Hej! Jag är 34 år gammal och en väldigt glad mäniiska!
Är du också glad? Det hoppas jag."""


def tokenize(text):
    """uses the nonletters to break the text into words
    returns a list of words"""
    # words = re.split('[\s\-,;:!?.’\'«»()–...&‘’“”*—]+', text)
    # words = re.split('[^a-zåàâäæçéèêëîïôöœßùûüÿA-ZÅÀÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ’\-]+', text)
    # words = re.split('\W+', text)
    words = re.split('\P{L}+', text)
    words.remove('')
    return words


def tokenize2(text):
    """uses the letters to break the text into words
    returns a list of words"""
    # words = re.findall('[a-zåàâäæçéèêëîïôöœßùûüÿA-ZÅÀÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ’\-]+', text)
    # words = re.findall('\w+', text)
    words = re.findall('\p{L}+', text)
    return words


def tokenize3(text):
    """uses the punctuation and nonletters to break the text into words
    returns a list of words"""
    # text = re.sub('[^a-zåàâäæçéèêëîïôöœßùûüÿA-ZÅÀÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ’'()\-,.?!:;]+', '\n', text)
    # text = re.sub('([,.?!:;)('-])', r'\n\1\n', text)
    text = re.sub(r'[^\p{L}\p{P}]+', '\n', text)
    text = re.sub(r'(\p{P})', r'\n\1\n', text)
    text = re.sub(r'\n+', '\n', text)
    return text.split()


def tokenize4(text):
    """uses the punctuation and symbols to break the text into words
    returns a list of words"""
    spaced_tokens = re.sub('([\p{S}\p{P}])', r' \1 ', text)
    one_token_per_line = re.sub('\s+', '\n', spaced_tokens)
    tokens = one_token_per_line.split()
    return tokens


if __name__ == '__main__':
    """words = tokenize(text)
    for word in words:
        print(word)
    words = tokenize2(text)
    print(words)"""
   # os.system('wc -w Selma.txt')
    words = tokenize(text)
    print(tokenize(text2))
    print(tokenize2(text2))
    print(tokenize3(text2))
    print(tokenize4(text2))
    count = 0
    count1 = 0
    for word in words:
        count1 += 1
        if word == 'gick':
            count += 1
    print(count)
    print(count1)
