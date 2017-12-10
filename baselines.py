import csv
import urllib
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as stopper
import time
from nltk.stem.snowball import SnowballStemmer
import math
import numpy as np
from lsa import *
from cosinesim import *
import copy

class Baselines:

	def __init__(self, corpus, wCorpus):

		self.TAG_RE = re.compile(r'<[^>]+>')
		self.WUP_THRESH = 0.4
		self.MAX_EDIT_DISTANCE = 3

		self.nouns = ['NN','NNS','NNP','NNPS'] # NN.*
		self.verbs = ['VB','VBD','VBG','VBN','VBP','VBZ'] # VB.
		self.numbers = ['CD']
		self.corpus=corpus
		self.wCorpus=wCorpus

	def text2int(self, textnum, numwords={}):
		if not numwords:
		  units = [
			"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
			"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
			"sixteen", "seventeen", "eighteen", "nineteen",
		  ]

		  tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

		  scales = ["hundred", "thousand", "million", "billion", "trillion"]

		  numwords["and"] = (1, 0)
		  for idx, word in enumerate(units):	numwords[word] = (1, idx)
		  for idx, word in enumerate(tens):	 numwords[word] = (1, idx * 10)
		  for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

		current = result = 0
		for word in textnum.split():
			if word not in numwords:
			  if 'inf' in word.lower():
				return 1234567 #Till we deal with infinity case
			  continue

			scale, increment = numwords[word]
			current = current * scale + increment
			if scale > 100:
				result += current
				current = 0

		return result + current

	def is_number(self, s):
		try:
			float(s)
			return True
		except ValueError:
			return False


	def remove_tags(self, text):
		return self.TAG_RE.sub('', text)
		

	class POSSets:
		def __init__(self):
			self.nouns = set()
			self.verbs = set()
			self.numbers = set()
			self.others = set()


	def getSets(self, pt):
		s = self.POSSets()
		for w in pt:
			if w[1] in self.nouns:
				s.nouns.add(w[0])
			elif w[1] in self.verbs:
				s.verbs.add(w[0])
			elif w[1] in self.numbers:
				s.numbers.add(w[0])
			else:
				s.others.add(w[0])
		return s

	def editDistance(self, mat,i,j,w1,w2):
		if(mat[i][j] == -1 ):
			if i==0:
				mat[i][j] = j
			elif j==0:
				mat[i][j] = i
			else:
				mat[i][j] = min(self.editDistance(mat,i-1,j,w1,w2)+1,self.editDistance(mat,i,j-1,w1,w2)+1, (self.editDistance(mat,i-1,j-1,w1,w2) if w1[i]==w2[j] else self.editDistance(mat,i-1,j-1,w1,w2)+2 ) )
		return mat[i][j]



	def bestSim(self, w1,w2):
		maxscore = 0
		#If any of the two words are not in wordNet, perform edit distance
		if not wn.synsets(w1) or not wn.synsets(w2):
			mat = [[-1 for x in range(len(w2)+1)] for x in range(len(w1)+1)] 
			if self.editDistance(mat,len(w1),len(w2),"#"+w1,"#"+w2) > self.MAX_EDIT_DISTANCE:
				return 0
			else:
				return 1
		
		for i in wn.synsets(w1,pos= wn.NOUN):
			for j in wn.synsets(w2,pos = wn.NOUN):
				score = i.wup_similarity(j)
				if score>maxscore:
					maxscore = score
		return maxscore

	def myround(self, x, base=10):
		return math.ceil(float(x)*base) /base

	def nmlcs1(self, a,b):
		for i in xrange(min(len(a),len(b))):
			if not a[i] == b[i]:
				break
		return a[:i]
		
	def pmiir(self, w1, w2):  #Uses the simplest equation for PMI-IR
		w1H=self.hitsInCorpus(w1)
		w2H=self.hitsInCorpus(w2)
		w12H=self.hitsInCorpus(w1, w2)

		if w1H>0 and w2H>0 and w12H>0:
			return math.log(float(w12H)/(float(w1H)*float(w2H)), 2) #hits(w1, w2)/hits(w1)
		return 0.0
		
		
	def commonWordOrder(self, W1,W2):

		if len(W1)==1:
			return 1
		if len(W1)==0:
			return 0
		div = len(W1)**2
		if len(W1)%2 == 1:
			div -= 1
		s = 0
		for i,w in enumerate(W1):
			s += abs(i-W1.index(W2[i]))
		s0 = 1.0 - ((2.0*s)/div)
		return s0
		
	def hitsInCorpus(self, w1, w2=None):  #Calculates number of documents w1 appears in. If w2 is provided, calculates bigram count instead.
		retCount=0.0
		for it in self.wCorpus:
			if w1 in it and ((not w2) or (w2 in it)):
				retCount+=1.0
		return retCount
		
	def lcs(self, a, b):
		lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
		# row 0 and column 0 are initialized to 0 already
		for i, x in enumerate(a):
			for j, y in enumerate(b):
				if x == y:
					lengths[i+1][j+1] = lengths[i][j] + 1
				else:
					lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
		# read the substring out from the matrix
		result = ""
		x, y = len(a), len(b)
		while x != 0 and y != 0:
			if lengths[x][y] == lengths[x-1][y]:
				x -= 1
			elif lengths[x][y] == lengths[x][y-1]:
				y -= 1
			else:
				assert a[x-1] == b[y-1]
				result = a[x-1] + result
				x -= 1
				y -= 1
		return result[::-1]
		
	def nmlcsn(self, a, b):
		m = [[0] * (1 + len(b)) for i in xrange(1 + len(a))]
		longest, x_longest = 0, 0
		for x in xrange(1, 1 + len(a)):
			for y in xrange(1, 1 + len(b)):
				if a[x - 1] == b[y - 1]:
					m[x][y] = m[x - 1][y - 1] + 1
					if m[x][y] > longest:
						longest = m[x][y]
						x_longest = x
				else:
					m[x][y] = 0
		return a[x_longest - longest: x_longest]

	def calculateMax(self, matrix):
		maxV=0
		maxI=0
		maxJ=0
		for it in range(len(matrix)):
			for jt in range(len(matrix[it])):
				if matrix[it][jt]>maxV:
					maxV=matrix[it][jt]
					maxI=it
					maxJ=jt
		return maxV, maxI, maxJ

	def baseLine1(self, sent1, sent2):
		
		#line = line.split(',')
		actualAnswer = sent1
		studentAnswer = sent2
		pt1 = nltk.pos_tag(nltk.word_tokenize(actualAnswer))
		pt2 = nltk.pos_tag(nltk.word_tokenize(studentAnswer))
					
		s1 = self.getSets(pt1)
		s2 = self.getSets(pt2)
		
		#nounsq = list(qs.nouns)
		nouns1= list(s1.nouns)
		nouns2= list(s2.nouns)
				
		naiveScore = 0
		
		#Compare nouns
		
		for i,ni in enumerate(nouns1):
			word1= ""
			word2= ""
			maxVal= 0.0
			for j,nj in enumerate(nouns2):
				if(nj.lower() == ni.lower()):
					naiveScore += 1
					break
				
				tmp= self.bestSim(nouns1[i],nouns2[j])
				
				if tmp>self.WUP_THRESH:
					naiveScore += tmp
					break
		
		#Compare numbers

		n1 = {self.text2int(x.lower())  if not self.is_number(x) else float(x) for x in s1.numbers}
		n2 = {self.text2int(x.lower())  if not self.is_number(x) else float(x) for x in s2.numbers}
		
		numMatches = n1 & n2
		numScore = len(numMatches)
		
		if (len(nouns1)+len(n1) > 0):
			naiveScore = float(naiveScore+numScore)/(len(nouns1)+len(n1))
		else:
			naiveScore = 0.0

		return naiveScore

	def baseLine2(self, sent1, sent2):
		stopwords=stopper.words('english')
		ignorechars = ''',:'!'''

		#ans = line.split(',')
		actualAnswer = sent1
		studentAnswer = sent2
		pt1 = nltk.word_tokenize(actualAnswer)
		pt2 = nltk.word_tokenize(studentAnswer)
		if(pt2==0):
			return 0
	
		if len(pt1)>len(pt2):
			pt1, pt2= pt2, pt1
		W1=[]
		temp=copy.deepcopy(pt2)
		for x in pt1:
			if x in pt2:
				del pt2[pt2.index(x)]
				W1.append(x)
		W2=[]
		for x in temp:
			if x in pt1:
				del pt1[pt1.index(x)]
				W2.append(x)

		m = len(pt1)
		n = len(pt2)

		s0 = self.commonWordOrder(W1,W2)
		delta = len(W1)
		wf = 0.05
		#Removing common words
		pt1 = [x for x in pt1 if x not in W1 ]
		pt2 = [x for x in pt2 if x not in W2 ]
	
		w = 1.0/3
		pmiirList=[]
		mat = [[-1 for x in range(len(pt2))] for y in range(len(pt1))] 
		for i,w1 in enumerate(pt1):
			for j,w2 in enumerate(pt2):
				prod =  len(w1)*len(w2)
				v1 = float(len(self.lcs(w1,w2))**2) / prod
				v2 = float(len(self.nmlcsn(w1,w2))**2) / prod
				v3 = float(len(self.nmlcs1(w1,w2))**2) / prod
				mat[i][j]  = w*(v1+v2+v3)

		matrix=mat
		rho=[]
		for it in range(len(matrix)):
			maxV, maxI, maxJ= self.calculateMax(matrix)
			rho.append(maxV)
			for jt in range(len(matrix[0])):
				matrix[maxI][jt]=-1
			for jt in range(len(matrix)):
				matrix[jt][maxJ]=-1

		m = m+delta
		n = n+delta
		SPR=float(((delta*(1-wf+(wf*s0)))+sum(rho))*(m+n))/(2*m*n)

		return SPR
		
		
	#LSA
	def baseLine3(self, sent1, sent2):
	
		alpha= 0.5
			
	#	ans = line.split(',')
		actualAnswer = sent1
		studentAnswer = sent2

		mylsa = LSA()
		mylsa.parse(actualAnswer)
		mylsa.parse(studentAnswer)
		mylsa.build()
		if mylsa.calcSVD()== 0:
			sim=0.0
		else:
			sim= float(mylsa.calcSimilarity())

		return sim

	#Cosine similarity
	def baseLine4(self, sent1, sent2):
	

		#ans = line.split(',')
		actualAnswer = sent1
		studentAnswer = sent2
		cosineSim=float(get_cosine(text_to_vector(actualAnswer), text_to_vector(studentAnswer)))

		return cosineSim

	def baseLine5(self, sent1, sent2):
		
	#	ans = line.split(',')
		actualAnswer = sent1
		studentAnswer = sent2
		pt1=nltk.word_tokenize(actualAnswer)
		pt2=nltk.word_tokenize(studentAnswer)
		if(pt2==0):
			return 0
		if len(pt1)>len(pt2):
			pt1, pt2= pt2, pt1
		W1=[]
		temp=copy.deepcopy(pt2)
		for x in pt1:
			if x in pt2:
				del pt2[pt2.index(x)]
				W1.append(x)
		W2=[]
		for x in temp:
			if x in pt1:
				del pt1[pt1.index(x)]
				W2.append(x)
	   
		m = len(pt1)
		n = len(pt2)
		delta = len(W1)
		pt1 = [x for x in pt1 if x not in W1 ]
		pt2 = [x for x in pt2 if x not in W2 ]
		matrix = [[-1 for x in range(len(pt2))] for y in range(len(pt1))] 
		for i,w1 in enumerate(pt1):
			for j,w2 in enumerate(pt2):
				matrix[i][j]=self.pmiir(w1, w2)
		rho=[]

		for it in range(len(matrix)):
			maxV, maxI, maxJ= self.calculateMax(matrix)
			rho.append(maxV)
			for jt in range(len(matrix[0])):
				matrix[maxI][jt]=-1
			for jt in range(len(matrix)):
				matrix[jt][maxJ]=-1	

		m = m+delta
		n = n+delta
		SPR=float((delta+sum(rho))*(m+n))/(2*m*n)
		return SPR	  

	def baseLine6(self, sent1, sent2):
		
	#	ans = line.split(',')
		actualAnswer = sent1
		studentAnswer = sent2
		pt1=nltk.word_tokenize(actualAnswer)
		pt2=nltk.word_tokenize(studentAnswer)
		if(pt2==0):
			return 0
		if len(pt1)>len(pt2):
			pt1, pt2= pt2, pt1
		W1=[]
		temp=copy.deepcopy(pt2)
		for x in pt1:
			if x in pt2:
				del pt2[pt2.index(x)]
				W1.append(x)
		W2=[]
		for x in temp:
			if x in pt1:
				del pt1[pt1.index(x)]
				W2.append(x)
	   
		m = len(pt1)
		n = len(pt2)
		delta = len(W1)
		pt1 = [x for x in pt1 if x not in W1 ]
		pt2 = [x for x in pt2 if x not in W2 ]
		matrix = [[-1 for x in range(len(pt2))] for y in range(len(pt1))] 
		for i,w1 in enumerate(pt1):
			for j,w2 in enumerate(pt2):
				matrix[i][j]=self.bestSim(w1,w2)
				#print w1 + " " + w2 + " = " + str(matrix[i][j])
		rho=[]
		for it in range(len(matrix)):
			maxV, maxI, maxJ= self.calculateMax(matrix)
			rho.append(maxV)
			for jt in range(len(matrix[0])):
				matrix[maxI][jt]=-1
			for jt in range(len(matrix)):
				matrix[jt][maxJ]=-1
		m = m+delta
		n = n+delta
		SPR=float((delta+sum(rho))*(m+n))/(2*m*n)
	 
		return SPR	  


	def out(self, line):

		ans = line.split(',')
		totalMarks=float(ans[-2])
		return float(ans[-1])/totalMarks
