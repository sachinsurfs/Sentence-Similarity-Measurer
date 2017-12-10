import re, math
from collections import Counter
import numpy
from gensim import *
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text, re.I)
     return Counter(words)

# def length(s):
#      val=0.0
#      for it in s:
#           val+=s**2
#      val=val**0.5
#      return val

# def cosineSim(s1, s2):
#      s1=sent2vec(s1)
#      s2=sent2vec(s2)
#      return numpy.dot(s1, s2)/(length(s1)*length(s2))