
"""
Concordance program to find all the concordances
of a pattern surrounded by width characters.
Usage: python concord.py file pattern width
"""
__author__ = "Pierre Nugues"

import re
import sys

file_name = 'Selma.txt'
#pattern = 'Nils'
width = 11
try:
    file = open(file_name)
except:
    print("Could not open file", file_name)
    exit(0)

text = file.read()
testText = "Jag vet inte att det ska vara så. Men det kan inte vara så svårt att säga nej! Eller är det de?"

"""
Tokenizers
Usage: python tokenizer.py < corpus.txt
"""

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


def setTag(words):
    tagedList = []
    pattern = re.compile("[\.\?\!]")
    start = True
    for word in words:
        tagedWord = word
        if pattern.match(word) and not start:
            start = True
            tagedWord = "</s>"
        elif word.istitle() and start:
            start = False
            tagedList.append("<s>")
        tagedList.append(tagedWord.lower())
    return tagedList


if __name__ == '__main__':
    words = tokenize4(text)
    words1 = setTag(words)
    print(words1)
