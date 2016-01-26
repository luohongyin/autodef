#lhy
#2016.1

from nltk.corpus import wordnet as wn

class Dictionary:
	def __init__(self):
		self.generator = wn.all_synsets()

	def get_words(self):
		l = [sn for sn in self.generator]
		self.wordList = list(set(map((lambda x:x.name().split('.')[0]), l)))

	def get_info(self, word):
		sn = wn.synsets(word)
		l = []
		for s in sn:
			if s.name().split('.')[0] != word:
				break
			l.append({"pos":s.pos(), "def":self.parse_sen(s.definition())})
		return l

	def get_all(self, word):
		sn = wn.synsets(word)
		return map((lambda x:{"name":x.name().split('.')[0], "pos":x.pos(), "def":self.parse_sen(x.definition())}), sn)

	def parse_sen(self, s):
		l = list(s)
		return ''.join(map((lambda x:x if x.isalpha() else ''), l))