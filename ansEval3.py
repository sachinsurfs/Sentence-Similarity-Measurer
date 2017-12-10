from stripKeywords import *
import pickle
import numpy as np
from baselines import Baselines


MAX_MARKS = 2
question = "Question : Why are protocols needed?"
print question
keywordsQ = nltk.word_tokenize(stripWords(question,[]))

ansCorpus = ["In networks, communication occurs between the entities in different systems. Two entities cannot just send bit streams to each other and expect to  be understood. For communication, the entities must agree on a protocol. A protocol is a set of rules that govern data communication." , "If computers did not have this set of rules, they would not have the capability of communicating over networks. Certain protocols help computers identify themselves across networks and most importantly the internet.", "Network protocols were created to allow computers to communicate in an organized manner without any room for misinterpretation. Clients that do not follow the rules oftentimes are disconnected by the server, or vice versa, depending on what the protocol specifications state.", "protocols allow computers to communicate with other computers without users having to know what is happening in the background."]
ans = raw_input("Ans : ")
actualAns = ansCorpus[0]#raw_input()#"ActualAns : ")

ansCorpuskw = []
for i in ansCorpus:
	ansCorpuskw.append(stripWords(i,keywordsQ))
	
wCorpus = []
for it in range(len(ansCorpuskw)):
	 wCorpus.append(nltk.word_tokenize(ansCorpuskw[it]))
anskw = stripWords(ans,keywordsQ)
actualAnskw = stripWords(actualAns,keywordsQ)

#ansCorpuskw22 = "".join(ansCorpuskw) 

#print "Student's answer: "+anskw
#print "Teacher's answer: "+actualAnskw

b = Baselines(ansCorpuskw,wCorpus)
BaselineResults = []
maxscores = [0.0]*5 
for iii in ansCorpuskw:
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
print "Pointwise Mutual Information - Information Retrieval : "+str(maxscores[3])
print "----"
print "Logistic Regression Results"
print "Recommended marks: "+str(float(skResults.predict(np.asarray(maxscores).reshape(1, -1))[0])*MAX_MARKS)
 



#for br in BaselineResults:
#	print float(skResults.predict(np.asarray(br).reshape(1, -1))[0])*MAX_MARKS

#for br in BaselineResults:
#	print br

#print keywordsQ
