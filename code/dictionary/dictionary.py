#lhy
#2016.1

import json
import parser

class Dictionary:
	def __init__(self, route):
		self.route = route
		self.dictionary = {}

	def build_dictionary(self):
		table = [chr(i) for i in range(97,123)]
		for w in table:
			p = parser.Parser("%sCIDE.%s" % (self.route, w.upper()))
			d = p.parse_file()
			for (key, value) in d.items():
				self.dictionary[key] = value
		json.dump(self.dictionary, open("../../data/dictionary.json", 'wb'))
		return self.dictionary