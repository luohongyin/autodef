local THNN = require 'nn.THNN'
local LMLookupTable, parent = torch.class('nn.LMLookupTable', 'nn.Module')

LMLookupTable.__version = 4

function LMLookupTable:__init(route, paddingValue)
   parent.__init(self)
   self.matrix = torch.load(route)
   self.weight = self.matrix
   self.gradWeight = torch.Tensor():resizeAs(self.matrix):zero()
   self.paddingValue = paddingValue or 0

   self:reset()
end

function LMLookupTable:backCompatibility()
   self._count = self._count or torch.IntTensor()
   self._input = self._input or torch.LongTensor()

   if not self.shouldScaleGradByFreq then
      self.shouldScaleGradByFreq = false
   end
end

function LMLookupTable:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function LMLookupTable:setPadding(paddingValue)
    self.paddingValue = paddingValue
    return self
end

function LMLookupTable:scaleGradByFreq()
   self.shouldScaleGradByFreq = true
   return self
end

function LMLookupTable:reset(stdv)
   stdv = stdv or 1
   self.weight:normal(0, stdv)
end

function LMLookupTable:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

function LMLookupTable:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.matrix, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end
   return self.output
end

function LMLookupTable:accGradParameters(input, gradOutput, scale)
   self:backCompatibility()
   input = self.copiedInput and self._input or input
   if input:dim() == 2 then
      input = input:view(-1)
   elseif input:dim() ~= 1 then
      error("input must be a vector or matrix")
   end

   if not gradOutput:isContiguous() then
       self._gradOutput = self._gradOutput or gradOutput.new()
       self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
       gradOutput = self._gradOutput
   end

   self.gradWeight.THNN.LookupTable_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self._count:cdata(),
      THNN.optionalTensor(self._sorted),
      THNN.optionalTensor(self._indices),
      self.shouldScaleGradByFreq or false,
      self.paddingValue or 0,
      scale or 1
   )
end

function LMLookupTable:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.matrix.new()
      self._indices = self.matrix.new()
      self._count = self.matrix.new()
      self._input = self.matrix.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end

function LMLookupTable:clearState()
   self._gradOutput = nil
   return self
end

-- we do not need to accumulate parameters when sharing
LMLookupTable.sharedAccUpdateGradParameters = LMLookupTable.accUpdateGradParameters
