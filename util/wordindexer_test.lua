--[[
Word Indexer Test

Woohyun Kim(deepcoord@gmail.com)
--]]

require("parser")
require("filereader")
require("wordindexer")

cmd = torch.CmdLine()
cmd:option("-file", 'file', 'file to index words')

opt = cmd:parse(arg)

if opt.file == "" then
  print("Usage: th wordindexer_test.lua -file <file>")
  os.exit()
end

if not paths.filep(opt.file) then
  print("Cannot find " .. opt.file)
  print("Usage: th wordindexer_test.lua -file <file>")
  os.exit()
end

tokens = {}
tokens.SEP = "%s" -- word separator
tokens.EOP = "<eop>" -- end of a pair
tokens.EOS = "<eos>" -- end of a sentence
tokens.UNK = "<unk>" -- unkown word
-- char
tokens.ZEROPAD = ' ' -- zero-pad
tokens.START = '{' -- start of a word
tokens.END = '}' -- end of a word

freq = 1

local indexer = WordIndexer(tokens, freq)
indexer:add(opt.file)
indexer:save(string.gsub(opt.file, paths.extname(opt.file), "vocab"))
indexer:stats(true)
--print("#####################################")
--indexer:add("xxx/valid.txt")
--indexer:stats()
