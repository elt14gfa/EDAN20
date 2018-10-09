"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os
import operator
import pprint


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


def find_pairs(formatted_corpus):
    counter = 0

    sentences = []
    # Convert sentence lists to dicts
    for sentence in formatted_corpus:
        new_sentence = {}
        for word in sentence:
            id_n = word['id']
            new_sentence[id_n] = word
        sentences.append(new_sentence)
    pairs = {}
    for sentence in sentences:
        for word_id in sentence:
            word = sentence[word_id]
            print(word)
            if '-' not in word_id and word['deprel'] == SUBJ:
                verb_key = word['head']
                verb = str.lower(sentence[verb_key]['form'])
                subject = str.lower(word['form'])
                counter += 1
                pair = (subject, verb)
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1

    sorted_pairs = sorted(pairs.items(), key=operator.itemgetter(1), reverse=True)
    print('Number of pairs in corpus: ' + str(counter))
    print('Most frequent pairs: ')
    for i in range(0, 200):
        print(sorted_pairs[i])

def find_tripples(formatted_corpus):
    counter = 0

    sentences = []
    # Convert sentence lists to dicts
    for sentence in formatted_corpus:
        new_sentence = {}
        for word in sentence:
            id_n = word['id']
            new_sentence[id_n] = word
        sentences.append(new_sentence)
    tripples = {}
    for sentence in sentences:
        id_list =[]
        tmp_sentences = []
        tmp_sentence = {}
        tmp_tripple = {}
        for word_id in sentence:
            word = sentence[word_id]

            if '-' not in word_id and (word['deprel'] == SUBJ or word['deprel'] == OBJ):

                key = word['deprel']
                tmp_sentence[word_id] = word
                tmp_sentences.append(tmp_sentence)
                tmp_tripple[key] = tmp_sentence
                print(tmp_tripple[key][word_id]['head'])
                """
                verb_key = word['head']
                verb = str.lower(sentence[verb_key]['form'])
                subject = str.lower(word['form'])
                counter += 1
                tripple = (subject, verb)
                if tripple in tripples:
                    tripples[tripple] += 1
                else:
                    tripples[tripple] = 1
                """

                if id_list != None and tmp_tripple[SUBJ] != None and tmp_tripple[OBJ] != None:
                    for id in id_list:
                        if tmp_tripple[key][word_id]['head'] == tmp_tripple[OBJ][id]['head']:
                            print(word)


                id_list.append(word_id)






if __name__ == '__main__':

    MULTILINGUAL = False

    if not MULTILINGUAL:

        SUBJ = 'SS'
        OBJ = 'OO'

        column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

        train_file = 'hej/train.conll'
        # train_file = 'test_x'
        test_file = 'hej/test.conll'

        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_2006)
        #print(train_file, len(formatted_corpus))
        #print(formatted_corpus[0])
        find_pairs(formatted_corpus)
        #find_tripples(formatted_corpus)

    else:

        SUBJ = 'nsubj'
        OBJ = 'obj'

        """
        
        
        column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']
    
        files = get_files('hej','train.conll')
        for train_file in files:
            sentences = read_sentences(train_file)
            formatted_corpus = split_rows(sentences, column_names_u)
            print(train_file, len(formatted_corpus))
            print(formatted_corpus[0])
            
            find_pairs(formatted_corpus)
        
        """

