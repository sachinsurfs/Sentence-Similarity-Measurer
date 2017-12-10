from numpy import zeros
from scipy.linalg import svd
from math import *
from numpy import asarray, sum

class LSA(object):

	def __init__(self):

		self.stopwords = ['and','edition','for','in','little','of','the','to']
		self.ignorechars = ''',:'!'''
		self.wdict = {}
		self.dcount = 0	

	def parse(self, doc):

		words = doc.split();
		for w in words:
			w = w.lower().translate(None, self.ignorechars)
			if w in self.stopwords:
				continue
			elif w in self.wdict:
				self.wdict[w].append(self.dcount)
			else:
				self.wdict[w] = [self.dcount]
		self.dcount += 1	

	def build(self):

		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		self.A = zeros([len(self.keys), self.dcount])
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:
				self.A[i,d] += 1

	def calcSVD(self):

		try:
			self.U, self.S, self.Vt = svd(self.A)
		except:
			return 0

	def TFIDF(self):

		WordsPerDoc = sum(self.A, axis=0)		
		DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
		rows, cols = self.A.shape
		for i in range(rows):
			for j in range(cols):
				self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])

	def calcNorm2(self, vec):

		square= 0
		for i in range(len(vec)):
			square+= vec[i]**2

		return sqrt(square)


	def calcSimilarity(self):

		docMat= [list(k) for k in self.Vt]
		sumVal= 0

		for i in range(len(self.S)):
			docMat[i][0]*= self.S[i]
			docMat[i][1]*= self.S[i]

		#Cosine Similarity between two rows of Vt
		for i in range(len(docMat[0])):
			sumVal+= docMat[i][0]* docMat[i][1]

		tmp1= self.calcNorm2([docMat[0][0], docMat[1][0]])
		tmp2= self.calcNorm2([docMat[0][1], docMat[1][1]])

		try:
			docSim= sumVal/(tmp1*tmp2)
		except:
			docSim= 0.0

		if isnan(docSim):
			return 0.0

		return docSim
		

# mylsa = LSA(stopwords, ignorechars)
# for t in titles:
# 	mylsa.parse(t)
# mylsa.build()
# mylsa.calcSVD()
# print mylsa.calcSimilarity()