-- lhy
-- 2016.3

require "torch"
stringx = require "pl.stringx"

word_vectors = {}
for line in io.lines("../data/model/vectors.txt") do
	vector = {}
	w_v = stringx.split(line, ' ')
	for i = 2, #w_v do
		table.insert(vector, w_v[i])
	end
	table.insert(word_vectors, vector)
end

word_vectors = torch.Tensor(word_vectors)
torch.save("../data/model/vectors.t7", word_vectors:sub(2, -1))