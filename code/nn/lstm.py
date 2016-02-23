#lhy
#2016.2

import theano
import theano.tensor as T
import numpy as np

def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

class LSTM:
	def __init__(self, nin, c, h):
		self.input_gate = T.vector("input_gate")
		self.forget_gate = T.vector("forget_gate")
		self.output_gate = T.vector("output_Gate")
		self.cell = theano.shared(c)
		self.hidden_state = theano.shared(h)
		self.params = {}
		self.g_params = {}
		self.updates = []
		self.s = T.vector('s')
		self.w = T.vector('w')
		param = ['W', 'U', 'Z', 'b']
		parts = ["input", "forget", "cell", "output"]
		for part in parts:
			p = map((lambda x:"%s_%s" % (x, part)), param)
			for p_i in p:
				if p_i[0] == 'b':
					self.params[p_i] = theano.shared(np.zeros(nin), name = p_i)
					self.g_params[p_i] = T.vector("g_%s" % p_i)
				else:
					self.params[p_i] = theano.shared(ortho_weight(nin), name = p_i)
					self.g_params[p_i] = T.matrix("g_%s" % p_i)
				self.updates.append((self.params[p_i], self.params[p_i] - self.g_params[p_i]))
		self.input_gate = T.nnet.sigmoid(T.dot(self.params["W_input"], s) + \
			T.dot(self.params["U_input"], self.hidden_state) + \
			T.dot(self.params["Z_input"], w) + self.params["b_input"])
		self.forget_gate = T.nnet.sigmoid(T.dot(self.params["W_forget"], s) + \
			T.dot(self.params["U_forget"], self.hidden_state) + \
			T.dot(self.params["Z_forget"], w) + self.params["b_forget"])
		self.cell = self.forget_gate * self.cell + self.input_gate * T.tanh(T.nnet.sigmoid(T.dot(self.params["W_cell"], s) + \
			T.dot(self.params["U_cell"], self.hidden_state) + \
			T.dot(self.params["Z_cell"], w) + self.params["b_cell"]))
		self.output_gate = T.nnet.sigmoid(T.dot(self.params["W_output"], s) + \
			T.dot(self.params["U_output"], self.hidden_state) + \
			T.dot(self.params["Z_output"], w) + self.params["b_output"])
		self.hidden_state = self.output_gate * T.tanh(self.cell)
		self.forward = theano.function([self.s, self.w], self.hidden_state)

	def sgd(self, cost, rate):
		self.rate = theano.shared(rate, name = 'r')
		for p in self.params.keys():
			self.g_params[p] = self.rate * T.grad(cost, wrt = p)
		return theano.function([cost, self.rate], self.params, updates = self.updates)