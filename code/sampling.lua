local Sampling, Parent = torch.class('nn.Sampling', 'nn.Module')

function Sampling:__init(dim)
   Parent.__init(self)
   self.dim = dim
   self.train = true
   self.noise = torch.Tensor(1)
end

function Sampling:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      if self.dim == 1 then
         local size = input:size()[1]
         for i = 1, size do
            self.noise:bernoulli(input[i])
            self.output[i] = self.noise[1]
         end
      end
      elseif self.dim == 2 then
         local size = input:size()
         for i = 1, size[1] do
            for j = 1, size[2] do
               self.noise:bernoulli(input[i][j])
               self.output[i][j] = self.noise[1]
            end
         end
      end
   end
   elseif not self.v2 then
      self.output:mul(1-self.p)
   end
   self.mask = self.output
   return self.output
end

function Sampling:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   if self.train then
      if self.p > 0 then
         self.gradInput:cmul(self.mask) -- simply mask the gradients with the noise vector
      end
   else
      if not self.v2 and self.p > 0 then
         self.gradInput:mul(1-self.p)
      end
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