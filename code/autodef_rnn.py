# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

'''

from __future__ import print_function
import cPickle
import word2vec
import sys
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense
from keras.layers import recurrent
import numpy as np
from six.moves import range


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        def set_value(c, i):
            if c in self.char_indices and i < maxlen:
                X[i, self.char_indices[c]] = 1
            return 1
        update = map((lambda x, i:set_value(x, i)), C, range(len(C)))
        #for i, c in enumerate(C):
        #    if i > maxlen - 1:
        #        break
        #    try:
        #        X[i, self.char_indices[c]] = 1.0
        #    except:
        #        continue
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ' '.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
DIGITS = 3
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = 20
chars = cPickle.load(open(sys.argv[5], 'rb'))
vocab_size = len(chars)
ctable = CharacterTable(chars, MAXLEN)

questions = []
expected = []
seen = set()
model = word2vec.load(sys.argv[1])
word_list = cPickle.load(open(sys.argv[2], 'rb'))
def_list = cPickle.load(open(sys.argv[3], 'rb'))
TRAINING_SIZE = len(word_list)

print('Vectorization...')
X = np.zeros((len(word_list), int(sys.argv[4])), dtype=np.float32)
y = np.zeros((len(def_list), MAXLEN, vocab_size), dtype=np.float32)
for i, word in enumerate(word_list):
    X[i] = model[word]
print("Encoding...")
#y = np.array(map((lambda x:ctable.encode(x.split(' '), maxlen = MAXLEN)), def_list), dtype = np.bool)
y = np.array([ctable.encode(s.split(' '), maxlen = MAXLEN) for s in def_list])
#for i, sentence in enumerate(def_list):
#    y[i] = ctable.encode(sentence.split(' '), maxlen=MAXLEN)

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
#(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(X_train, X_val) = (X[:split_at], X[split_at:])
(y_train, y_val) = (y[:split_at], y[split_at:])
word_val = word_list[split_at:]
print(X_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(Dense(HIDDEN_SIZE, input_dim=400))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributedDense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 800):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val), show_accuracy=True, verbose = 0)
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = np.array([X_val[ind]]), y_val[ind]
        preds = model.predict_classes(rowX, verbose=0)
        q = word_val[ind]
        correct = ctable.decode(rowy)
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')

