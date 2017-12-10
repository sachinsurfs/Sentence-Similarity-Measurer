from stripKeywords import *
import pickle
import numpy as np

print "Question : What is software quality?"

MAX_MARKS = 1
ansCorpus = ["The degree to which a system, component, or process meets specified requirements.", "The extent to which a system, component, or process meets customer or user needs or expectations.", "Conformance to requirements and fitness to use.", "Software quality may be defined as conformance to explicitly stated functional and performance requirements, explicitly documented development standards and implicit characteristics that are expected of all professionally developed software.","software quality measures how well software is designed (quality of design), and how well the software conforms to that design (quality of conformance).", "Software quality is the extent to which an industry-defined set of desirable features are incorporated into a product so as to enhance its lifetime performance.", "Quality is the totality of features and characteristics of a product or a service that bears on its ability to satisfy the given needs."]
from baselines import Baselines
ans = raw_input("Ans : ")
actualAns = raw_input("Actual Ans : ")#raw_input()#"ActualAns : ")

keywordsQ = ["software","quality"]
ansCorpuskw = []
for i in ansCorpus:
	ansCorpuskw.append(stripWords(i,keywordsQ))
	
wCorpus = []
for it in range(len(ansCorpuskw)):
	 wCorpus.append(nltk.word_tokenize(ansCorpuskw[it]))
anskw = stripWords(ans,keywordsQ)
actualAnskw = stripWords(actualAns,keywordsQ)

#ansCorpuskw22 = "".join(ansCorpuskw) 

#print "Student's answer: "+ans
#print "Teacher's answer: "+actualAns
b = Baselines(ansCorpuskw,wCorpus)
BaselineResults = []
maxscores = [0.0]*5 
for iii in [actualAnskw]:
	BaselineResults.append([b.baseLine2(anskw,iii),b.baseLine3(anskw,iii),b.baseLine4(anskw,iii),b.baseLine5(anskw,iii),b.baseLine6(anskw,iii)])
	for iv,IV in enumerate(BaselineResults[-1]):
		if IV>maxscores[iv]:
			maxscores[iv] = IV

#print b.baseLine1(anskw,ansCorpuskw22)

fp=open("savedRegressor_"+str(MAX_MARKS)+".pkl", "r")
skResults=pickle.load(fp)

print "Wu & Palmer: "+str(maxscores[4])
print "Lowest common substring + Common word order Similarity: "+str(maxscores[0])
print "Latent Semantic Analyis : "+str(maxscores[1])
print "Cosine Similarity : "+str(maxscores[2])
print "Pointwise Mutual Information - Information Retrieval : "+str(maxscores[2])
print "----"
print "Logistic Regression Results"
print "Recommended marks: "+str(float(skResults.predict(np.asarray(maxscores).reshape(1, -1))[0])*MAX_MARKS)
 

#print keywordsQ