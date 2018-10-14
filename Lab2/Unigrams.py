import regex as re
import math

def tokenizer(text):
    """breaks text within '.', '!' or '?' into new lines, putting each new line within '<s>' and '</s>' and removes
    all different kinds of .-/".!?. In addition it puts each word as its own element"""

    text = re.sub('\s+',' ', text)
    text = re.sub('([0-9\p{L}\"\:\,\- ]+[\.\!\?])',
                  '<s>' + r' \1' + ' </s>\n', text).lower()
    text = re.sub('([.\?\!\,\"\-])', "", text)
    text = re.sub(' +', ' ', text)
    words = re.findall('\\<[s/]*\\>|\w+',text)

    return words

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


def biunigrams(text,testWords):

    freq = count_ngrams(text,1)
    tempforBi = []
    sentences = {}
    prob = {}
    nbrWords = len(text)
    unigram_prob = 1
    entropy = 0

    print('Unigram model')
    for test in testWords:
        if test == '<s>':
            ''
        else:
            sentences[test] = 0
            decide = True
            for unigram in text:
                if test == unigram:
                    sentences[test] = freq[unigram,]
                    if decide is True:
                        tempforBi.append(sentences[test])
                        decide = False
            if sentences[test] == 0:
                tempforBi.append(sentences[test])

            prob[test] = sentences[test]/nbrWords
            unigram_prob *= prob[test]
            entropy += math.log(prob[test],2)
            print(test, '\t', sentences[test], '\t', nbrWords, '\t', prob[test])

    geo_prob = math.pow(unigram_prob, 1/((len(testWords))-1))
    entropyFinal = -entropy/(len(testWords)-1)

    print('===================================================')
    print('Prob. unigrams:' + ' ' + str(unigram_prob))
    print('Geometric mean prob:' + ' ' + str(geo_prob))
    print('Entropy rate:' + ' ' + str(entropyFinal))
    print('Perplexity:' + ' ' + str(math.pow(2, entropyFinal)))
    print('===================================================\n')
    print('Bigram Model')

    bifreqTest = count_ngrams(testWords, 2)
    bifreqText = count_ngrams(text, 2)
    bisentences = {}
    biprob = {}
    i=0
    N = len(testWords)
    bigram_prob = 1
    bientropy = 0

    for bitest in bifreqTest:
        bisentences[bitest] = 0
        i = i+1
        for bigram in bifreqText:

            if bitest == bigram:
                bisentences[bitest] = bifreqText[bigram]
        if tempforBi[i-2] != 0 and bisentences[bitest] != 0:
            biprob[bitest] = bisentences[bitest] / tempforBi[i-2]
        elif bisentences[bitest] == 0:
            biprob[bitest] = prob[bitest[1]]

        bigram_prob *= biprob[bitest]
        bientropy += math.log(biprob[bitest],2)

        print(bitest, '\t', bisentences[bitest], '\t', tempforBi[i-2], '\t', biprob[bitest])

    bigeo_prob = math.pow(bigram_prob, 1/(len(bifreqTest) - 1))
    bientropyFinal = -bientropy/(len(bifreqTest))

    print('===================================================')
    print('Prob. unigrams:' + ' ' + str(bigram_prob))
    print('Geometric mean prob:' + ' ' + str(bigeo_prob))
    print('Entropy rate:' + ' ' + str(bientropyFinal))
    print('Perplexity:' + ' ' + str(math.pow(2, bientropyFinal)))
    print('===================================================\n')


    return sentences, bisentences



if __name__ == '__main__':

    text = open('Selma.txt').read().lower()

    words = tokenizer(text)
    n=1
    frequency_ngrams = count_ngrams(words,n)
    testText = 'det var en gång en katt som hette nils.'
    testWords = tokenizer(testText)
    biunigrams(words,testWords)