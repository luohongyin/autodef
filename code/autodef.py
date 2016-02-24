#lhy
#2016.2

import word2vec
import theano
import theano.tensor as T
import numpy as np
from nn import lstm
from nn import neuroglia
from wordnet import dictionary
from collection import orderedDict
from collection import Counter

def get_corpus(route, num_common):
	inHandle = open(route, 'r')
	text = inHandle.read().split(' ')
	c = Counter(text)
	return map((lambda x:x[0]), c.most_common(num_common))

def main(nin, num_candidate, word_vector, num_iter, rate):
	d = dictionary.Dictionary()
	wordList = d.get_words()
	candidate = get_corpus(num_candidate)
	word_table = {}
	for i, word in enumerate(candidate):
		word_table[i] = word
	cell = np.random.rand(nin)
	hidden_state = np.zeros(nin)
	l_nn = lstm.LSTM(nin, cell, hidden_state)
	n_glia = neuroglia.Neuroglia(nin)
	model = word2vec.load(word_vector)
	W = theano.shared(np.random.rand(num_candidate, nin), name = 'W')
	L_s = theano.shared(np.random.rand(num_candidate, num_candidate), name = 'L_s')
	b = theano.shared(np.random.rand(num_candidate), name = 'b')
	for word in wordList:
		w = model[word]
		sentence = d.get_info(word)
		definition = ' '.join(map((lambda x:x["def"]), sentence)).split(' ')
		def_list = map((lambda x:np.array(map((lambda y:1.0 if word_table[y] == x else 0.0), range(num_candidate)))), definition)
		hidden_state = np.random.rand(nin)
		for i in range(num_iter):
			for i, def_i in enumerate(def_list):
				if i == 0:
					sample = n_glia.init_sample(w)
					vector = sample * w
					hidden_state = l_nn.forward(def_i, vector)
					prediction = T.nnet.sigmoid(T.dot(W, hidden_state) + b)
				else:
					sample = n_glia.sample(hidden_state)
					vector = sample * w
					hidden_state = l_nn.forward(def_i, vector)
					prediction = T.nnet.sigmoid(T.dot(W, hidden_state) + T.dot(L_s, def_list[i - 1]) + b)
				entropy = (-def_i * T.log(prediction) - (1 - def_i) * T.log(1 - prediction)).mean()
				cost = entropy + 0.01 * (w ** 2).sum()
				l_optimizer = l_nn.sgd(cost, rate)
				lstm_params = l_optimizer(cost, rate)
				n_optimizer = n_glia.sgd(cost, rate)
				nglia_params = n_optimizer(cost, rate)