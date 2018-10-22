import re
import os

dictionary = dict()
tfIdf = dict()
nbrOfFiles = 9

def index(text,filename):
    p = re.compile(r"\p{L}+")
    i = 0;
    for m in p.finditer(text.lower()):
        if (dictionary.get(m.group())== None):
            dictionary[(m.group())] = dict()
            tmplist = dictionary[m.group()]
            tmplist[filename] = [m.start()]
            dictionary[(m.group())] = tmplist
        elif(dictionary[m.group()].get(filename) == None):
            tmplist = dictionary[m.group()]
            tmplist[filename] = [m.start()]
            dictionary[(m.group())] = tmplist
        else:
            tmplist = dictionary[m.group()][filename]
            tmplist.append(m.start())
            dictionary[m.group()][filename] = tmplist

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

