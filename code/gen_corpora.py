#lhy
#2016.2

from wordnet import dictionary

d = dictionary.Dictionary()
words = d.get_words()
definitions = map((lambda x:d.get_info(x)[0]["def"]), words)
outHandle1 = open("model/words.txt", 'w')
outHandle2 = open("model/definitions.txt", 'w')
outHandle1.write('\n'.join(words))
outHandle2.write('\n'.join(definitions))
outHandle1.close()
outHandle2.close()