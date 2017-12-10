import nltk
import re
import csv
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

questions=[]	# List
answers=[]  # List
submissions=[] # List Of Lists

stemmer = SnowballStemmer("english")
stopper = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
nouns = ['NN','NNS','NNP','NNPS'] # NN.*

# Strip Keywords And Stopwords From All Answers and also lemmatize
def stripWords(sent, keywordsQ):
	sent = re.sub(r'[^\w\s]', '', sent)
	#re.sub("[^a-zA-Z0-9\.:;\?!,]","",sent)
	words=nltk.word_tokenize(sent)
	newWords = []
	#word_removal = []
	prev = ""
	for ii,word in enumerate(words):
		if not (wordnet_lemmatizer.lemmatize(word).lower().strip() in keywordsQ or word.lower().strip() in stopper):
			if not (word == prev):
				newWords.append(wordnet_lemmatizer.lemmatize(word).lower().strip())
			#re.sub(word, "", sent, flags=re.I)
			#word_removal.append(word)
		#if word == prev:
		#	words[ii] = ""
		prev = word
			#sent = sent.replace(word, "")
	'''
	for w in word_removal:
		try:
			words.remove(w)
		except:
			pass #Fix later
	'''
	#print words
	sent = " ".join(newWords)
	#sent = re.sub(" +"," ", sent)
	return str(sent)

# Get Keywords In Questions

if __name__=="__main__":
	filterWords = ["img","code","="]
	fp = open("finalData.sql", "r")
	#fp2 = open("SubjectiveAnswers_AfterKeywordRemoval.csv", "w")
	fp3 = open("myData.csv", "w")
	for line in fp:
		try:
			line = line.strip("\n").split("VALUES (")[1][:-3]
			for line in csv.reader([line], delimiter=',', quotechar='\''):
				question = line[7]
				answer = line[8]
				submission = line[9]
				keywordsQ=[]
				words=nltk.pos_tag(nltk.word_tokenize(question))
				for word, tag in words:
					if tag in nouns:	# Anything Which Is A Noun
						keywordsQ.append(stemmer.stem(word).lower())
							
				# Remove Keywords And Stopwords From Expected Answers And Actual Answers
				answer=stripWords(answer)
				submission=stripWords(submission)
				
				#fp2.write(line[0]+","+line[1]+","+line[2]+","+line[3]+","+line[4]+","+line[5]+","+line[6]+","+
				#question+","+answer+","+submission+","+line[10]+","+line[11]+","+line[12]+","+line[13]+ "\n")

				fp3.write("'"+question+"','"+answer+"','"+submission+"',"+line[11]+","+line[12]+ ",'" + line[2] + "'\n")
		except IndexError:
			pass #Skip line	

	fp.close()
	#fp2.close()
	fp3.close()