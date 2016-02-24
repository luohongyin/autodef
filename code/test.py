#lhy
#2016.2

import theano
import theano.tensor as T
import numpy as np
from nn import lstm
from wordnet import dictionary

l = lstm.LSTM()
l.init_param(200)
print "Init Param"

cell = np.random.rand(200)
hidden_state = np.random.rand(200)
l.init_state(cell, hidden_state)
s = T.vector('s')
w = T.vector('w')
forward = l.forward(s, w)
a = np.random.rand(200)
b = np.random.rand(200)
c, h = forward(a, b)
print c
c, h = forward(a, c)
print c