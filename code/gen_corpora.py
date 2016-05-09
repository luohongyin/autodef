#lhy
#2016.2

import sys
import word2vec
import numpy as np
from wordnet import dictionary

model = word2vec.load(sys.argv[1])

def get_vector(word):
	global model
	if '_' in word:
		wordList = word.split('_')
		word_num = len(wordList)
		v = np.zeros(int(sys.argv[2]))
		for w in wordList:
			try:
				v += model[w]
			except:
				word_num -= 1
		if word_num == 0:
			return ''
		return "%s %s\n" % (word, ' '.join(map((lambda x:str(x)), v / word_num)))
	try:
		v = model[word]
		return "%s %s\n" % (word, ' '.join(map((lambda x:str(x)), v)))
	except:
		return ''

def select(w_v, s):
	if w_v == '':
		return ''
	return "%s\n" % s

d = dictionary.Dictionary()
words = d.get_words()
wordList = []
for word in words:
    if len(d.get_info(word)) != 0 and len(word) != 0:
        wordList.append(word)
definitions = map((lambda x:d.get_info(x)[0]["def"]), wordList)
word_vectors = map(get_vector, wordList)
wordList = map(select, word_vectors, wordList)
definitions = map(select, word_vectors, definitions)
word_num = len(word_vectors) - word_vectors.count('')
wvHandle = open("../data/model/vectors.txt", 'w')
wvHandle.write("%s %s\n" % (word_num, sys.argv[2]))
wvHandle.write(''.join(word_vectors))
wvHandle.close()
outHandle1 = open("../data/model/words.txt", 'w')
outHandle2 = open("../data/model/definitions.txt", 'w')
outHandle1.write(''.join(wordList))
outHandle2.write(''.join(definitions))
outHandle1.close()
outHandle2.close()
