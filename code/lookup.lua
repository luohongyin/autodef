-- Use word vectors as the weight of LookupTable
local Lookup, Parent = torch.class('neuralconvo.Lookup', 'nn.Module')

function Lookup:__init(route)
   self.weight = torch.load(route)
end

function Lookup:updateOutput(input)
   self.output = self.weight:index(1, torch.LongTensor(torch.totable(input)))
   return self.output
end

function Lookup:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end

function Lookup:clearState()
   return Parent.clearState(self)
end