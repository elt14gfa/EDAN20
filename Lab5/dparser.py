"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features
import numpy as np

feature_names_short = [
    'stack_0_word',
    'stack_0_POS',
    'queue_0_word',
    'queue_0_POS',
    'can-re',
    'can-la'
]

feature_names_middle = [
    'stack_0_word',
    'stack_0_POS',
    'stack_1_word',
    'stack_1_POS',
    'queue_0_word',
    'queue_0_POS',
    'queue_1_word',
    'queue_1_POS',
    'can-re',
    'can-la'
]

feature_names_long = [
    'stack_0_POS',
    'stack_0_word',
    'stack_1_word',
    'stack_1_POS',
    'queue_0_word',
    'queue_0_POS',
    'queue_1_word',
    'queue_1_POS',
    'after_stack_0_word',
    'after_stack_0_POS',
    'can-re',
    'can-la',
    'can-ra'
]


def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

def all_features(formatted_corpus):
    y = []  #list with all actions
    X_tot = []
    a = 0
    sent_cnt = 0
    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            pass

        stack = []
        queue = list(sentence)
        # print('queue:', queue)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []

        while queue:
            X = []  # matrix for all featuresc
            x = features.extract(stack, queue, graph, feature_names_middle, sentence)
            for token in x:
                X.append(x[token])
                X_tot.append(x[token])

            #print('X =', X, end = " ")
            stack, queue, graph, trans = reference(stack, queue, graph)
            y.append(trans)
            #print('y =', y[a], '\n')

        X_tot.append('\n')

        a += 1



        stack, graph = transition.empty_stack(stack, graph)


    return X_tot, y





if __name__ == '__main__':
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    #feat, action = (all_features(formatted_corpus))

    X, y = all_features(formatted_corpus)

    print('X:', X)
    #print('y:', y)

    sent_cnt = 0
    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            #print(sent_cnt, 'sentences on', len(formatted_corpus))
            flush = True
        stack = []
        queue = list(sentence)
        #print('queue:', queue)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []
        while queue:
            stack, queue, graph, trans = reference(stack, queue, graph)
            transitions.append(trans)
            feat = features.extract(stack, queue, graph, feature_names_middle, sentence)


        stack, graph = transition.empty_stack(stack, graph)
        #print('Equal graphs:', transition.equal_graphs(sentence, graph))
        list_word = []
        i = 0
        # Poorman's projectivization to have well-formed graphs.
        #print('sentence:', sentence)

        for word in sentence:
            list_word.append(word['form'])
            word['head'] = graph['heads'][word['id']]
        #print(transitions)
        #print(graph)
