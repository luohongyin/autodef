-- lhy --
-- 2016.3 --

local WordNet = torch.class("neuralconvo.WordNet")
local stringx = require "pl.stringx"
local xlua = require "xlua"

function WordNet:__init(dir)
	self.dir = dir
end

local TOTAL_WORDS = 72499

local function process(c)
	if c % 10000 == 0 do
		xlua.process(c, TOTAL_WORDS)
	end
end

function WordNet:load()
	lines = {}
	count = 0
	for line in io.input(self.dir .. "/definitions.txt") do
		table.insert(lines, line)
		count = count + 1
		process(count)
	end
	return lines
end