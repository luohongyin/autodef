local Lookup, Parent = torch.class('nn.Lookup', 'nn.Module')

function Lookup:__init(route)
   self.weight = torch.load(route)
end

function Lookup:updateOutput(input)
   self.output = self.weight:index(1, torch.LongTensor(torch.totable(input)))
   return self.output
end

function Lookup:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   return self.gradInput
end

function Lookup:setp(p)
   self.p = p
end

function Lookup:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end

function Lookup:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end
