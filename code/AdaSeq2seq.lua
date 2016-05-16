-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local AdaSeq2Seq = torch.class("neuralconvo.AdaSeq2Seq")

function AdaSeq2Seq:__init(vocabSize, hiddenSize)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")

  self:buildModel()
end

function AdaSeq2Seq:buildModel()
  -- self.encoder = nn.Sequential()
  -- self.encoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  -- self.encoder:add(nn.SplitTable(1, 2))
  -- self.encoderLinear = nn.Linear(400, self.hiddenSize)
  -- self.encoder:add(nn.Linear(self.encoderLinear)
  -- self.encoder:add(nn.SelectTable(-1))

  local para = nn.ParallelTable()
  local lookupModule = nn.Sequential()
  local linearModule = nn.Sequential()
  local samplingModule = nn.Sequential()
  self.LMModule = nn.Sequential()
  self.MEMModule = nn.Sequential()
  local attentionModule = nn.Sequential()
  lookupModule:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  
  --[[
  self.encoderLSTM = nn.LSTM(self.hiddenSize, 400)
  samplingModule:add(nn.SplitTable(1, 2))
  samplingModule:add(nn.Sequencer(self.encoderLSTM))
  samplingModule:add(nn.Sequencer(nn.Reshape(1, 400)))
  samplingModule:add(nn.JoinTable(1))
  samplingModule:add(nn.Linear(400, 400))
  samplingModule:add(nn.Sigmoid())
  samplingModule:add(nn.Sampling(2))
  ]]--
  linearModule:add(nn.Linear(400, self.hiddenSize))
  para:add(linearModule):add(lookupModule)
  self.decoder = nn.Sequential()
  self.decoder:add(para)
  --[[
  LMModule:add(nn.Dropout(0))
  concat = nn.ConcatTable()
  concat:add(samplingModule):add(LMModule)

  lookupModule:add(concat)
  para:add(linearModule):add(lookupModule)
  self.decoder = nn.Sequential()
  self.decoder:add(para)
  self.decoder:add(nn.FlattenTable())
  concat2 = nn.ConcatTable()
  attentionModule:add(nn.NarrowTable(1, 2))
  attentionModule:add(nn.CMulTable())
  attentionModule:add(nn.Linear(400, self.hiddenSize))
  concat2:add(attentionModule):add(nn.SelectTable(3))
  self.decoder:add(concat2)
  ]]--
  self.decoder:add(nn.CAddTable())
  self.decoder:add(nn.SplitTable(1, 2))
  self.decoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.LMModule:add(self.decoder)
  self.LMModule:add(nn.Sequencer(nn.Linear(self.hiddenSize, self.vocabSize)))
  self.LMModule:add(nn.Sequencer(nn.LogSoftMax()))
  self.LMModule:zeroGradParameters()
  --[[
  self.decoder2 = self.decoder:clone('weight', 'bias')
  self.MEMModule:add(self.decoder2)
  self.MEMModule:add(nn.JoinTable(1))
  self.MEMModule:zeroGradParameters()
  ]]--
  self.zeroTensor = torch.Tensor(2):zero()
end

function AdaSeq2Seq:cuda()
  -- self.encoder:cuda()
  self.LMModule:cuda()
  --self.MEMModule:cuda()
  if self.criterion then
    self.criterion:cuda()
  end
  --[[
  if self.MEMCriterion then
	self.MEMCriterion:cuda()
  end
  ]]--

  self.zeroTensor = self.zeroTensor:cuda()
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function AdaSeq2Seq:forwardConnect(inputSeqLen)
  self.decoderLSTM.userPrevOutput =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLinear.output)
  self.decoderLSTM.userPrevCell =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLinear.output)
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function AdaSeq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell =
    nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
end

function AdaSeq2Seq:train(input, target)
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  -- self.encoder:forward(encoderInput)
  -- self:forwardConnect(encoderInput:size(1))
  local LMModelOutput = self.LMModule:forward({input, decoderInput})
  --local MEMOutput = self.MEMModule:forward({input, decoderInput})
  local Edecoder = self.criterion:forward(LMModelOutput, decoderTarget)

  if Edecoder ~= Edecoder then -- Exist early on bad error
    return Edecoder
  end

  -- Backward pass
  local gEdec = self.criterion:backward(LMModelOutput, decoderTarget)
  --[[
  local len = #LMModelOutput
  local inputTable = torch.Tensor(len, 400):cuda()
  for i = 1, len do
    inputTable[i] = input[1]
  end
  local mEdec = self.MEMCriterion:backward(MEMOutput, inputTable)
  ]]--
  self.LMModule:backward({input, decoderInput}, gEdec)
  -- self:backwardConnect()
  -- self.encoder:backward(encoderInput, self.zeroTensor)
  -- self.encoder:updateGradParameters(self.momentum)
  self.LMModule:updateGradParameters(self.momentum)
  self.LMModule:updateParameters(self.learningRate)
  -- self.encoder:updateParameters(self.learningRate)
  -- self.encoder:zeroGradParameters()
  self.LMModule:zeroGradParameters()

  -- self.encoder:forget()
  --[[
  self.MEMModule:backward({input, decoderInput}, mEdec)
  self.MEMModule:updateGradParameters(self.momentum)
  self.MEMModule:updateParameters(self.learningRate)
  self.MEMModule:zeroGradParameters()
  self.MEMModule:forget()
  ]]--
  self.LMModule:forget()
  return Edecoder
end

local MAX_OUTPUT_SIZE = 30

function AdaSeq2Seq:eval(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  -- self.encoder:forward(input)
  -- self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = self.goToken
  for i = 1, MAX_OUTPUT_SIZE do
    local prediction = self.LMModule:forward({input, torch.Tensor{output}})[1]
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

  self.LMModule:forget()
  -- self.encoder:forget()

  return predictions, probabilities
end
