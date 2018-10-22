import pprint
import regex as re
import math
import scipy
import numpy as np
p = re.compile(r"\p{L}+")

dic = dict()
dic1 = dict()
list1 = list()

text = 'aa bb cc dd aa 22 ff hh hh gsdfgf gdrgrs aa aa ff hh aa aa aa'
for m in p.finditer(text.lower()):
    if dic.get(m.group()) is None:
        dic.update({m.group(): m.start()})
    else:
        dic[m.group()] = [dic[m.group()], m.start()]


pprint.pprint(dic)








