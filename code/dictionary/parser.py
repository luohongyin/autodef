#lhy
#2016.1

import re

def select_alpha(w):
	if w.isalpha():
		return w
	else:
		return ''

class Parser:
    def __init__(self, filename):
        self.filename = filename
        self.inHandle = open(filename, 'r')
        self.text = self.inHandle.read()

    def parse(self):
    	r = re.compile("<hw>([^<>]*)?</hw>.*?<def>(.*)?</def>")
    	self.text_list = r.findall(self.text)
    	#print self.text_list

    def get_word(self, w):
    	l = ''.join(map(select_alpha, list(w))).lower()
    	return l

    def get_sentence(self, s):
    	l = list(s)
    	signal = 0
    	for i in range(len(l)):
    		if l[i] == '<':
    			signal = 1
    		if l[i] == '>':
    			signal = 0
    		if (not l[i].isalpha() or signal == 1) and l[i] != ' ':
    			l[i] = ''
    	sentence = ''.join(l).lower()
    	return sentence

    def parse_file(self):
    	self.parse()
    	dictionary = {}
    	for item in self.text_list:
    		dictionary[self.get_word(item[0])] = self.get_sentence(item[1])
    	return dictionary