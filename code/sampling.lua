require "cutorch"
require "cunn"

local Sampling, Parent = torch.class('nn.Sampling', 'nn.Module')

function Sampling:__init(dim)
   Parent.__init(self)
   self.dim = dim
   self.noise = torch.Tensor()
end

function Sampling:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.mask = torch.rand(input:size()):cuda()
   self.output = torch.clamp(torch.sign(input - self.mask), 0, 1)
   self.noise = self.output
   return self.output
end

function Sampling:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   if self.train then
      self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   end
   return self.gradInput
end

function Sampling:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end

function Sampling:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end
