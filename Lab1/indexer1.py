import os
import pickle
import regex as re
import math
import pprint


def create_index(textFile):
    dic = dict()
    file = open(textFile, 'r')
    text = file.read()
    p = re.compile(r"\p{L}+")

    for m in p.finditer(text.lower()):
        if dic.get(m.group()) is None:
            dic[m.group()] = [m.start() + 1]
        else:
            dic[m.group()].append(m.start() + 1)

    newname = textFile.split(".")
    newname[-1] = "idx"
    pickle.dump(dic, open(".".join(newname), "wb"))
    return dic


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def create_master_index(dir):
    txtfiles = get_files('Selma', 'txt')
    masterdic = dict()

    for file in txtfiles:
        create_index(dir + "/" + file)

    idxFiles = get_files('Selma', 'idx')
    idxFiles1 = ['bannlyst.idx', 'gosta.idx']
    for file in idxFiles:
        idxdic = pickle.load(open(dir + "/" + file, "rb"))
        for word in idxdic:
            if word not in masterdic:
                masterdic[word] = {file: idxdic[word]}
            else:
                masterdic[word][file] = idxdic[word]

    pickle.dump(masterdic, open('master_index.idx', 'wb'))

    return masterdic


"TF method should first see which texts contains the chosen words" \
"Then the documents should be distinguished by looking at the frequency of which the words appears in a text and the length of the document" \
"IDF method is important when the chosen words are very commonly used. IDF will then reduce the weight factor for that word and in increaes it for rare words" \
"TF-IDF is the product of the two factors described above"

def get_nbrOfWords(idxfiles, dir):
    nbrOfWords = {}
    for file in idxfiles:
        nbrOfWords[file] = 0
        idxdic = pickle.load(open(dir + "/" + file, "rb"))
        for word in idxdic:
            if idxdic[word] is not None:
                nbrOfWords[file] += (len(idxdic[word]))
    return (nbrOfWords)


"""
def calc_tf_idf(master_index,idxfiles,dir):
    doc_count = len(idxfiles)
    nbrWords = get_nbrOfWords(idxfiles,dir)
    word_array = []
    corpus_arrays = {}
    for filename in idxfiles:
        corpus_arrays[filename] = {}
    for word, word_index in master_index.items():

        # Calc idf
        nr_docs_containing_word = len(word_index)
        idf = math.log10(doc_count / nr_docs_containing_word)

        for idxfile in idxfiles:
            #calc TF
            occurences = word_index[idxfile] if idxfile in word_index else []
            tf = len(occurences) / nbrWords[idxfile]
            word_array[word] = tf * idf
            corpus_arrays[idxfile].apppend(word_array[word])
            print(idxfile,corpus_arrays[idxfile])
    return corpus_arrays
"""


def get_tf_idf(word, dir, master_index):
    tfidfdic = {}
    idxfiles = get_files('Selma', 'idx')
    nbrWords = get_nbrOfWords(idxfiles, dir)
    doc_count = len(idxfiles)
    for file in idxfiles:
        try:
            tf = len(master_index[word][file]) / nbrWords[file]
            idf = math.log10(doc_count / len(master_index[word]))
            tfidfdic[file] = {word: tf * idf}
        except:
            tfidfdic[file] = {word: 0}

    return tfidfdic


def get_tf_idf_AllwordsOneFile(idxfile, dir, master_index):
    wordtfidf = {}
    idxfiles = get_files('Selma', 'idx')
    nbrWords = get_nbrOfWords(idxfiles, dir)
    doc_count = len(idxfiles)
    for word in master_index:
        try:
            tf = len(master_index[word][idxfile]) / nbrWords[idxfile]
            idf = math.log10(doc_count / len(master_index[word]))
            wordtfidf[word] = tf * idf
        except:
            wordtfidf[word] = 0

    return wordtfidf


def cosSimilarities(idxfile1, idxfile2, dir, master_index):
    idtflist1 = get_tf_idf_AllwordsOneFile(idxfile1, dir, master_index)
    idtflist2 = get_tf_idf_AllwordsOneFile(idxfile2, dir, master_index)
    sumd1times2 = 0
    sumd1sqr = 0
    sumd2sqr = 0
    for word in idtflist1:
        sumd1times2 += idtflist1[word] * idtflist2[word]
        sumd1sqr += idtflist1[word] * idtflist1[word]
    for word in idtflist2:
        sumd2sqr += idtflist2[word] * idtflist2[word]

    return sumd1times2 / (math.sqrt(sumd1sqr) * math.sqrt(sumd2sqr))


"""
Start main
"""
#print(create_index('Selma/bannlyst.txt'))
masterindex = pickle.load(open('master_index.idx', 'rb'))


#pprint.pprint(create_master_index('Selma'))

cosValues = {}

filenames = get_files('Selma','idx')
i = 0

for filename1 in filenames:
    for filename2 in filenames:
        if filename1 != filename2:
            cosValues[filename1,filename2] = cosSimilarities(filename1,filename2,'Selma',masterindex)
            if cosValues[filename1,filename2] > i:
                i = cosValues[filename1,filename2]
                tmp1 = str(filename1)
                tmp2 = str(filename2)
                tmp3 = str(i)

pprint.pprint(cosValues)
print("Best texts:" + tmp1 + tmp2 + ":" + tmp3)




# create_master_index('Selma')
# files = get_files('Selma', 'idx')

# get_tf_idf_AllwordsOneFile('bannlyst.idx','Selma',masterindex)
# pprint.pprint(masterindex)
#pprint.pprint(get_tf_idf('nils', 'Selma', masterindex))

# pprint.pprint(get_tf_idf_AllwordsOneFile('bannlyst.idx','Selma',masterindex))
