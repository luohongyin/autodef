#lhy
#2016.2

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

class Neuroglia:
	def __init__(self, nin):
		self.T = theano.shared(np.random.rand(nin, nin), name = 'T')
		self.Q = theano.shared(float(nin), name = 'Q')
		self.U = theano.shared(np.random.rand(nin, nin), name = 'U')
		self.H = theano.shared(np.random.rand(nin, nin), name = 'H')
		self.T_sum = theano.shared(np.ones((nin, nin)), name = 'Ts')
		self.h_sum = theano.shared(np.zeros(nin), name = 'hs')
		self.w = T.vector('w')
		self.h = T.vector('h')
		self.G = T.exp(T.dot(self.H, self.w))
		self.E = T.exp(self.G)
		self.P = self.E / self.E.sum()
		self.h_updated = self.h + self.h_sum
		self.T_updated = T.dot(self.T, self.T_sum)
		self.G_updated = self.G + T.dot(self.U, self.h_updated)
		self.Q_updated = T.exp(self.G_updated).sum()
		self.P_updated = T.dot(self.T_updated, self.P) * T.exp(T.dot(self.U, self.h_updated)) * self.Q / self.Q_updated
		self.srng = RandomStreams(seed=234)
		self.rv_u = srng.uniform((nin,))
		self.init_s = (self.P > self.rv_u).astype("float")
		self.s = (self.P_updated > self.rv_u).astype("float")
		self.init_sample = theano.function(self.w, self.init_s)
		self.sample = theano.function(self.h, self.s, updates = [(self.h_sum, self.h_updated), (self.T_sum, self.T_updated)])

	def sgd(self, cost, rate):
		self.rate = theano.shared(rate, name = 'r')
		g_T, g_U, g_H = self.rate * T.grad(cost, wrt = ['T', 'U', 'H'])
		return theano.function([cost, self.rate], [g_T, g_U, g_H], \
			updates = [(self.T, self.T - g_T), (self.U, self.U - g_U), (self.H, self.H - g_H)])