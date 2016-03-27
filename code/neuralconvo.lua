require 'torch'
require 'nn'
require 'rnn'

neuralconvo = {}

torch.include('neuralconvo', 'wordnet.lua')
torch.include('neuralconvo', 'dataset.lua')
torch.include('neuralconvo', 'movie_script_parser.lua')
torch.include('neuralconvo', 'AdaSeq2seq.lua')
torch.include('neuralconvo', 'sampling.lua')
torch.include('neuralconvo', 'seq2seq.lua')

return neuralconvo
