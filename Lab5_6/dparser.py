"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features
import pickle
import numpy as np
import pprint
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
import time
from sklearn import tree
from sklearn.model_selection import cross_val_score

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

FEATURE_NAMES = feature_names_long


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
    y = []  # list with all actions
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
            # matrix for all featuresc
            x = features.extract(stack, queue, graph, FEATURE_NAMES, sentence)
            X_tot.append(x)
            # X_tot.append(x[token])

            # print('X =', X, end = " ")

            stack, queue, graph, trans = reference(stack, queue, graph)
            #print('stack:', stack, '\n', 'trans:', trans[:2])
            #if(stack == 'ra'):
            #    print('\n', 'stack is ra:', stack)
            y.append(trans)
            # print(', y =', y[a], '\n')

        a += 1

        stack, graph = transition.empty_stack(stack, graph)

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]

    return X_tot, y


def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    if stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    if stack and trans == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'




if __name__ == '__main__':
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    MODEL = 'LOGISTIC'

    if MODEL == 'LOGISTIC':
        classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
   # elif MODEL == 'PERCEPTRON':
    #    classifier = linear_model.Perceptron(penalty='l2')
   # elif MODEL == 'DECISION':
    #    classifier = tree.DecisionTreeClassifier()


    print("Extracting the features...")
    X_dict, y_train_symbols = all_features(formatted_corpus)
    print("Encoding the features...")
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)



    try:
        if FEATURE_NAMES == feature_names_short:
            model = pickle.load(open('predicted_model_short.txt', 'rb'))

        elif FEATURE_NAMES == feature_names_middle:
            model = pickle.load(open('predicted_model.txt', 'rb'))

        else:
            model = pickle.load(open('predicted_model_long.txt', 'rb'))

        print(model)
        print("The model was loaded from memory, skipping training phase")
    except:

        print("Training the model...")

        model = classifier.fit(X, y_train_symbols)

        y_train_predicted = classifier.predict(X)
        print(model)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_train_symbols, y_train_predicted)))

        if FEATURE_NAMES == feature_names_short:
            pickle.dump(model, open('predicted_model_short.txt', 'wb'))

        elif FEATURE_NAMES == feature_names_middle:
            pickle.dump(model, open('predicted_model.txt', 'wb'))

        else:
            pickle.dump(model, open('predicted_model_long.txt', 'wb'))



    sentences = conll.read_sentences(test_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sent_cnt = 0

    y_predicted_symbols = []  # Our array of transistions

    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            a = 1
            # print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        while queue:

            x = features.extract(stack, queue, graph, FEATURE_NAMES, sentence)
            X_test = vec.transform(x)
            trans_predict = model.predict(X_test)[0]
            stack, queue, graph, trans = parse_ml(stack, queue, graph, trans_predict)

            # Save the predicted trans
            y_predicted_symbols.append(trans)

        stack, graph = transition.empty_stack(stack, graph)

    #for i in range(0,100):
     #   print('y_train:', y_train_symbols[i], '\t', 'y_predict:', y_predicted_symbols[i])

        #print(graph)
        for word in sentence:
            word_id = word['id']
            try:
                word['head'] = graph['heads'][word_id]
                word['phead'] = graph['heads'][word_id]
            except KeyError:
                word['head'] = '_'
                word['phead'] = '_'

            try:
                word['deprel'] = graph['deprels'][word_id]
                word['pdeprel'] = graph['deprels'][word_id]
            except KeyError:
                word['deprel'] = '_'
                word['pdeprel'] = '_'
        #print(sentence)


    conll.save('results_long.txt', formatted_corpus, column_names_2006)



"""

if __name__ == '__main__':
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    LOGISTIC = 'logistic-regression'
    PERCEPTRON = 'perceptron'
    DECISION = 'decision-tree-classifier'

    MODEL = LOGISTIC

    if MODEL == LOGISTIC:
        classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    elif MODEL == PERCEPTRON:
        classifier = linear_model.Perceptron(penalty='l2')
    elif MODEL == DECISION:
        classifier = tree.DecisionTreeClassifier()

    X_dict, y = all_features(formatted_corpus)
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)

    try:
        # Model was found
        model = pickle.load(open(MODEL, 'rb'))

        print("The model was loaded from memory, skipping training phase")

    except FileNotFoundError:
        print("Extracting the features...")

        # print(X_dict)
        print("Encoding the features...")
        # Vectorize the feature matrix and carry out a one-hot encoding

        # The statement below will swallow a considerable memory
        # X = vec.fit_transform(X_dict).toarray()
        # print(vec.get_feature_names())

        training_start_time = time.clock()
        print("Training the model...")
        # classifier = DecisionTreeClassifier()
        # classifier = linear_model.perceptron(penalty='12')
        # classifier = Perceptron()
        model = classifier.fit(X, y)
        print(model)

        end_time = time.clock()
        print("Training time:", (training_start_time - end_time) / 60)

        pickle.dump(model, open(MODEL, 'wb'))

        X_test_dict, y_test = all_features(formatted_corpus)

        X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
        y_test_predicted = classifier.predict(X_test)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_test, y_test_predicted)))


        # Init test set
        sentences = conll.read_sentences(test_file)
        formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sent_cnt = 0

    y_predicted_symbols = []  # Our array of transistions

    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            a = 1
            # print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        while queue:
            x = features.extract(stack, queue, graph, FEATURE_NAMES, sentence)

            X_test = vec.transform(x)

            predicted_trans_index = model.predict(X_test)[0]
            predicted_trans = dict_classes[predicted_trans_index]

            # Build new graph
            stack, queue, graph, trans = execute_transition(stack, queue, graph, predicted_trans)

            # Save the predicted trans
            y_predicted_symbols.append(trans)

        stack, graph = transition.empty_stack(stack, graph)

        for word in sentence:
            word_id = word['id']
            try:
                word['head'] = graph['heads'][word_id]
                word['phead'] = graph['heads'][word_id]
            except KeyError:
                word['head'] = '_'
                word['phead'] = '_'

            try:
                word['deprel'] = graph['deprels'][word_id]
                word['pdeprel'] = graph['deprels'][word_id]
            except KeyError:
                word['deprel'] = '_'
                word['pdeprel'] = '_'

    conll.save('results.txt', formatted_corpus, column_names_2006)

    # We apply the model to the test set
    test_sentences = list(conll.read_sentences(test_file))
    test_formatted_corpus = conll.split_rows(test_sentences, column_names_2006_test)


    X_test_dict, y_test = all_features(test_formatted_corpus)

    X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
    y_test_predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_test_predicted)))

    print("Predicting the test set...")
    f_out1 = open('out1', 'w')
    predict(test_sentences, feature_names, f_out1)
    

    

    # feat, action = (all_features(formatted_corpus))
    
    X, y = all_features(formatted_corpus)
    print(y)

    #print('X:', X)
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
        i = 0
        # Poorman's projectivization to have well-formed graphs.
        #print('sentence:', sentence)

        for word in sentence:
            word['head'] = graph['heads'][word['id']]
        #print(transitions)
        #print(graph)
"""