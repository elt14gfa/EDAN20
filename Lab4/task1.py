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
    for i in range(0, 5):
        try:
            print(sorted_pairs[i])
        except:
            print('No more pairs')

def find_triplesWrong(formatted_corpus):
    counter = 0

    sentences = []
    # Convert sentence lists to dicts
    for sentence in formatted_corpus:
        new_sentence = {}
        for word in sentence:
            id_n = word['id']
            new_sentence[id_n] = word
        sentences.append(new_sentence)

    triples = {}
    for sentence in sentences:
        print('new sentence')
        tmp_obj = {}
        tmp_subj = {}
        for word_id in sentence:
            word = sentence[word_id]
            # print(word)
            if '-' not in word_id and (word['deprel'] == SUBJ or word['deprel'] == OBJ):
                # print('found ' + word['deprel'])
                key = word['deprel']
                if key == SUBJ:
                    tmp_subj[word_id] = word
                else:
                    tmp_obj[word_id] = word
                if tmp_subj is not None and tmp_obj is not None:
                    for subj_id in tmp_subj:
                        for obj_id in tmp_obj:
                            if tmp_subj[subj_id]['head'] == tmp_obj[obj_id]['head']:
                                verb_key = word['head']
                                verb = str.lower(sentence[verb_key]['form'])
                                counter += 1
                                triple = (tmp_subj[subj_id]['form'], verb, tmp_obj[obj_id]['form'])
                                if triple in triples:
                                    triples[triple] += 1
                                else:
                                    triples[triple] = 1

                """if tmp_subj is not None and tmp_obj is not None and tmp_subj['head'] == tmp_obj['head']:
                    verb_key = word['head']
                    verb = str.lower(sentence[verb_key]['form'])
                    counter += 1
                    triple = (tmp_subj['form'], verb, tmp_obj['form'])
                    # print('This is the triple: ')
                    # print(triple)
                    if triple in triples:
                        triples[triple] += 1
                    else:
                        triples[triple] = 1
    """
    sorted_triples = sorted(triples.items(), key=operator.itemgetter(1), reverse=True)
    print('Number of triples in corpus: ' + str(counter))
    print('Most frequent triples: ')
    for i in range(0, 5):
        print(sorted_triples[i])

def find_triples(formatted_corpus):
    counter = 0

    sentences = []
    # Convert sentence lists to dicts
    for sentence in formatted_corpus:
        new_sentence = {}
        for word in sentence:
            id_n = word['id']
            new_sentence[id_n] = word
        sentences.append(new_sentence)

    triples = {}
    for sentence in sentences:
        id_subj = []
        id_obj = []
        tmp_subj = {}
        tmp_obj = {}
        for word_id in sentence:
            word = sentence[word_id]
            if '-' not in word_id and (word['deprel'] == SUBJ or word['deprel'] == OBJ):
                key = word['deprel']

                if id_obj is not None and id_subj is not None:

                    if key == SUBJ:
                        tmp_subj[word_id] = word
                        for id in id_obj:
                            if tmp_subj[word_id]['head'] == tmp_obj[id]['head']:
                                counter += 1
                                verb_key = word['head']
                                verb = (str.lower(sentence[verb_key]['form']))
                                subj = (str.lower(tmp_subj[word_id]['form']))
                                obj = (str.lower(tmp_obj[id]['form']))
                                triple = (subj, verb, obj)
                                if triple in triples:
                                    triples[triple] += 1
                                else:
                                    triples[triple] = 1
                        id_subj.append(word_id)


                    elif key == OBJ:

                        tmp_obj[word_id] = word
                        for id in id_subj:
                            if tmp_subj[id]['head'] == tmp_obj[word_id]['head']:
                                counter += 1
                                verb_key = word['head']
                                verb = str.lower(sentence[verb_key]['form'])
                                subj = str.lower(tmp_subj[id]['form'])
                                obj = str.lower(tmp_obj[word_id]['form'])
                                triple = (subj, verb, obj)
                                if triple in triples:
                                    triples[triple] += 1
                                else:
                                    triples[triple] = 1
                        id_obj.append(word_id)

    sorted_triples = sorted(triples.items(), key=operator.itemgetter(1), reverse=True)
    print('Number of triples in corpus: ' + str(counter))
    print('Most frequent triples: ')
    for i in range(0, 5):
        try:
            print(sorted_triples[i])
        except:
            print('No more triples')
            break

if __name__ == '__main__':

    MULTILINGUAL = False

    if MULTILINGUAL:

        SUBJ = 'nsubj'
        OBJ = 'obj'

        column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']
        files = get_files('dependencies/ud-treebanks-v2.2', 'train.conllu')
        for train_file in files:
            if train_file == 'dependencies/ud-treebanks-v2.2/UD_Swedish-Talbanken/sv_talbanken-ud-train.conllu'\
                    or train_file == 'dependencies/ud-treebanks-v2.2/UD_English-ParTUT/en_partut-ud-train.conllu' \
                    or train_file == 'dependencies/ud-treebanks-v2.2/UD_Danish-DDT/da_ddt-ud-train.conllu':
                sentences = read_sentences(train_file)
                formatted_corpus = split_rows(sentences, column_names_u)
                # print(train_file, len(formatted_corpus))
                # find_pairs(formatted_corpus)
                find_triples(formatted_corpus)
    else:

        SUBJ = 'SS'
        OBJ = 'OO'

        column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
        train_file = 'train.conll'
        test_file = 'test.conll'

        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_2006)
        print(train_file, len(formatted_corpus))
        find_pairs(formatted_corpus)
        find_triples(formatted_corpus)
