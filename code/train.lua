require 'neuralconvo'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'approximate size of dataset to use (0 = all)')
cmd:option('--minWordFreq', 1, 'minimum frequency of words kept in vocab')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--hiddenSize', 1000, 'number of hidden units in LSTM')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.000025, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 400, 'number of examples to load at once')

cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
  options.dataset = nil
end

-- Data
print("-- Loading dataset")
dataset = neuralconvo.DataSet(neuralconvo.WordNet("../data/model"),
                    {
                      loadFirst = options.dataset,
                      minWordFreq = options.minWordFreq
                    })

print("\nDataset stats:")
print("  Vocabulary size: " .. dataset.wordsCount)
print("         Examples: " .. dataset.examplesCount)

-- Model
model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = options.learningRate
model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
local minMeanError = nil

-- Enabled CUDA
if options.cuda then
  require 'cutorch'
  require 'cunn'
  model:cuda()
end

-- Run the experiment
for epoch = 1, options.maxEpoch do
  local test_examples = {}
  local test_id = {}
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")
  local errors = torch.Tensor(dataset.examplesCount):fill(0)
  local timer = torch.Timer()
  local i = 1
  for examples in dataset:batches(options.batchSize) do
    collectgarbage()

    for _, example in ipairs(examples) do
      local input, target = unpack(example)
      if i % 1000 == 0 then
		    vector = torch.Tensor(1, 400)
		    vector[1] = example[1]
        table.insert(test_examples, {vector, example[2]})
		    table.insert(test_id, i)
      end

	    if i % 1000 ~= 0 then
	      local encoderInput = torch.Tensor(target:size()[1] - 1, 400)
	      for i = 1, target:size()[1] - 1 do
	        encoderInput[i] = input
	      end

        if options.cuda then
          encoderInput = encoderInput:cuda()
          target = target:cuda()
        end

        local err = model:train(encoderInput, target)

        -- Check if error is NaN. If so, it's probably a bug.
        if err ~= err then
          error("Invalid error! Exiting.")
        end

        errors[i] = err
	    end
      -- xlua.progress(i, dataset.examplesCount)
      i = i + 1
    end
  end

  timer:stop()

  print("\nFinished in " .. xlua.formatTime(timer:time().real) .. " " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
  print("\nEpoch stats:")
  print("           LR= " .. model.learningRate)
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. errors:mean())
  print("          std= " .. errors:std())

  -- Save the model if it improved.
  if minMeanError == nil or errors:mean() < minMeanError then
    print("\n(Saving model ...)")
    torch.save("../data/model/model.t7", model)
    minMeanError = errors:mean()
  end

  model.learningRate = model.learningRate + decayFactor
  model.learningRate = math.max(options.minLR, model.learningRate)

  -- Load testing script
  require "eval"

  for i = 1, #test_examples do
    print("Test Example " .. i)
    say(test_examples[i][1])
	print(test_id[i])
    print("----------------------------------------")
  end
end
