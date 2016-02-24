#lhy
#2016.2

from wordnet import dictionary

d = dictionary.Dictionary()
words = d.get_words()
wordList = []
for word in words:
    if len(d.get_info(word)) != 0 and len(word) != 0:
        wordList.append(word)
definitions = map((lambda x:d.get_info(x)[0]["def"]), wordList)
outHandle1 = open("../data/model/words.txt", 'w')
outHandle2 = open("../data/model/definitions.txt", 'w')
outHandle1.write('\n'.join(wordList))
outHandle2.write('\n'.join(definitions))
outHandle1.close()
outHandle2.close()
