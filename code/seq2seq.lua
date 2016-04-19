-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")

  self:buildModel()
end

function Seq2Seq:buildModel()
  -- self.encoder = nn.Sequential()
  -- self.encoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  -- self.encoder:add(nn.SplitTable(1, 2))
  -- self.encoderLinear = nn.Linear(400, self.hiddenSize)
  -- self.encoder:add(nn.Linear(self.encoderLinear)
  -- self.encoder:add(nn.SelectTable(-1))

  -- local para = nn.ParallelTable()
  local lookupModule = nn.Sequential()
  -- local linearModule = nn.Sequential()
  lookupModule:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  -- lookupModule:add(nn.SplitTable(1, 2))
  -- linearModule:add(nn.Linear(400, self.hiddenSize))
  -- linearModule:add(nn.Dropout(0))
  -- para:add(linearModule):add(lookupModule)
  self.decoder = nn.Sequential()
  self.decoder:add(lookupModule)
  -- self.decoder:add(nn.CAddTable())
  self.decoder:add(nn.SplitTable(1, 2))
  self.decoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.Linear(self.hiddenSize, self.vocabSize)))
  self.decoder:add(nn.Sequencer(nn.LogSoftMax()))
  -- self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()
  self.zeroTensor = torch.Tensor(2):zero()
end

function Seq2Seq:cuda()
  -- self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end

  self.zeroTensor = self.zeroTensor:cuda()
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  self.decoderLSTM.userPrevOutput =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLinear.output)
  self.decoderLSTM.userPrevCell =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLinear.output)
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell =
    nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
end

function Seq2Seq:train(input, target)
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  -- self.encoder:forward(encoderInput)
  -- self:forwardConnect(encoderInput:size(1))
  self.decoderLSTM.userPrevOutput = input[1]
  local decoderOutput = self.decoder:forward(decoderInput)
  local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)

  if Edecoder ~= Edecoder then -- Exist early on bad error
    return Edecoder
  end

  -- Backward pass
  local gEdec = self.criterion:backward(decoderOutput, decoderTarget)
  self.decoder:backward(decoderInput, gEdec)
  -- self:backwardConnect()
  -- self.encoder:backward(encoderInput, self.zeroTensor)

  -- self.encoder:updateGradParameters(self.momentum)
  self.decoder:updateGradParameters(self.momentum)
  self.decoder:updateParameters(self.learningRate)
  -- self.encoder:updateParameters(self.learningRate)
  -- self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.decoder:forget()
  -- self.encoder:forget()

  return Edecoder
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  -- self.encoder:forward(input)
  -- self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}
  self.decoderLSTM.userPrevOutput = input

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = self.goToken
  for i = 1, MAX_OUTPUT_SIZE do
    local prediction = self.decoder:forward(torch.Tensor{output})[1]
    -- prediction contains the probabilities for each word IDs.
    -- The index of the probability is the word ID.
    local prob, wordIds = prediction:sort(1, true)

    -- First one is the most likely.
    output = wordIds[1]

    -- Terminate on EOS token
    if output == self.eosToken then
      break
    end
    table.insert(predictions, wordIds)
    table.insert(probabilities, prob)
  end 

  self.decoder:forget()
  -- self.encoder:forget()

  return predictions, probabilities
end
