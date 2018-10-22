import transition


def extract(stack, queue, graph, feature_names, sentence):

    features = []

    # stack_0
    if stack:
        stack_0_POS = stack[0]['postag']
        stack_0_word = stack[0]['form']
    else:
        stack_0_POS = 'nil'
        stack_0_word = 'nil'

    # queue_0
    if queue:
        queue_0_POS = queue[0]['postag']
        queue_0_word = queue[0]['form']
    else:
        queue_0_POS = 'nil'
        queue_0_word = 'nil'


    # stack_1
    if len(stack) > 1:
        stack_1_POS = stack[1]['postag']
        stack_1_word = stack[1]['form']
    else:
        stack_1_POS = 'nil'
        stack_1_word = 'nil'

    # queue_1
    if len(queue) > 1:
        queue_1_POS = queue[1]['postag']
        queue_1_word = queue[1]['form']
    else:
        queue_1_POS = 'nil'
        queue_1_word = 'nil'


    #Set1

    if len(feature_names) == 6:
        features.append(stack_0_POS)
        features.append(stack_0_word)
        features.append(queue_0_word)
        features.append(queue_0_POS)
        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    #Set2

    elif len(feature_names) == 10:

        features.append(stack_0_POS)
        features.append(stack_1_POS)
        features.append(stack_0_word)
        features.append(stack_1_word)
        features.append(queue_0_POS)
        features.append(queue_1_POS)
        features.append(queue_0_word)
        features.append(queue_1_word)


        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    elif len(feature_names) == 13:
        # word after top of stack in sentence
        if stack_0_word == 'nil':
            after_stack_0_word = 'nil'
            after_stack_0_POS = 'nil'
        else:
            id_stack_0 = int(stack[0]['id'])
            if len(sentence)-1 == id_stack_0: #stack 0 is the last word
                after_stack_0_word = 'nil'
                after_stack_0_POS = 'nil'
            else:
                next_word = sentence[id_stack_0+1]
                after_stack_0_word = next_word['form']
                after_stack_0_POS = next_word['postag']

        features.append(stack_0_POS)
        features.append(stack_1_POS)
        features.append(stack_0_word)
        features.append(stack_1_word)
        features.append(queue_0_POS)
        features.append(queue_1_POS)
        features.append(queue_0_word)
        features.append(queue_1_word)

        features.append(after_stack_0_word)
        features.append(after_stack_0_POS)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

        # Our own features
        features.append(transition.can_rightarc(stack))
        # features.append(head_of_stack_0_POS)

        # Convert features object


    features = dict(zip(feature_names, features))


    return features

