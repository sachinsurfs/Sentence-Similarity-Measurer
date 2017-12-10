from stripKeywords import *
import pickle
import numpy as np
from baselines import Baselines


MAX_MARKS = 3
question = "Question : Why is electron affinity of fluorine less than chlorine?"
print question
keywordsQ = nltk.word_tokenize(stripWords(question,[]))

ansCorpus = ["This makes the fluoride anion so formed unstable (highly reactive) due to a very high charge/mass ratio. Also, fluorine has no d-orbitals, which limits its atomic size. As a result, fluorine has an electron affinity less than that of chlorine.", "Fluorine, though higher than chlorine in the periodic table, has a very small atomic size. This makes the fluoride anion so formed unstable (highly reactive) due to a very high charge/mass ratio. Also, fluorine has no d-orbitals, which limits its atomic size. As a result, fluorine has an electron affinity less than that of chlorine.", "The reason that the electron affinity is not as high as might otherwise be predicted for fluorine is that it is an extremely small atom, and so it's electron density is very high. Adding an additional electron is therefore not quite as favorable as for an element like chlorine where the electron density is slightly lower (due to electron-electron repulsion between the added electron and the other electrons in the electron cloud).", "Because of its small size, the fluorine atom exerts a significant pull on its electrons. However, its small size also causes the electrons to approach each other very closely, meaning they start to repel each other due to their like charge. To insert another electron in the highly repelling environment of fluorine would take a certain amount of energy. Chlorine is larger in size. Because of this, it is less reactive, since the nucleus does not exert such a great pull on electrons. However, inserting an electron in a shell where there is much more space will not take as much energy, because the repulsion to be overcome is lower. This means that it is more energy efficient than for the fluorine atom. Hence, chlorine has a higher electron affinity than fluorine."]
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
print "MAX MARKS -"

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
 

#for br in BaselineResults:
#	print float(skResults.predict(np.asarray(br).reshape(1, -1))[0])*MAX_MARKS

#for br in BaselineResults:
#	print br

#print keywordsQ
