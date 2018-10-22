
import conll_reader

train_corpus = 'train.txt'
test_corpus = 'test.txt'
w_size = 2
sentence = conll_reader.read_sentences(train_corpus)

start = "BOS BOS BOS\n"
end = "\nEOS EOS EOS"
start *= w_size
end *= w_size
sentence = start + sentence
sentence += end

print(sentence)