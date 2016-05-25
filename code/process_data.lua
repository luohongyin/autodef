-- lhy
-- 2016.3

require "torch"
stringx = require "pl.stringx"

word_vectors = {}
n = 0
for line in io.lines("../data/model/vectors.txt") do
	if n == 1 then
		vector = {}
		w_v = stringx.split(line, ' ')
		for i = 2, #w_v do
			table.insert(vector, tonumber(w_v[i]))
		end
		table.insert(word_vectors, vector)
	end
	if n == 0 then
		n = 1
	end
end

word_vectors = torch.Tensor(word_vectors)
torch.save("../data/model/vectors.t7", word_vectors)
