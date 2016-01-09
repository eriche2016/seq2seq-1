--[[
Input Indexer Test

Woohyun Kim(deepcoord@gmail.com)
--]]

require("parser")
require("filereader")
require("wordindexer")
require("inputloader")

local path = require('pl.path')

cmd = torch.CmdLine()
cmd:option('-data_dir','data', 'data directory containing train.txt/valid.txt/test.txt')
cmd:option('-min_freq', 5, 'minimum frequences for building vocabulary')

opt = cmd:parse(arg)

if not path.exists(opt.data_dir) then
  print("Usage: th inputloader_test.lua -data_dir <data_dir>")
  os.exit()
end

-- input files
local train_file = path.join(opt.data_dir, 'train.txt')
local valid_file = path.join(opt.data_dir, 'valid.txt')
local test_file = path.join(opt.data_dir, 'test.txt')

if not paths.filep(train_file) then print("Cannot find " .. train_file); os.exit() end
if not paths.filep(valid_file) then print("Cannot find " .. valid_file); os.exit() end
if not paths.filep(test_file) then print("Cannot find " .. test_file); os.exit() end

tokens = {}
tokens.SEP = "%s" -- word separator
tokens.EOP = "<eop>" -- end of a pair
tokens.EOS = "<eos>" -- end of a sentence
tokens.UNK = "<unk>" -- unkown word
-- char
tokens.ZEROPAD = ' ' -- zero-pad
tokens.START = '{' -- start of a word
tokens.END = '}' -- end of a word

-- build up vocabularies from train/valid/test
local indexer = WordIndexer(tokens, opt.min_freq)
indexer:add(train_file)
indexer:add(valid_file)
indexer:add(test_file)
indexer:stats()

-- load files with the given vocabularies
local keep = false
local reverse = true
local loader = InputLoader(indexer)

-- save word-wise file
local train_data = loader:load(train_file)
loader:save(string.gsub(train_file, paths.extname(train_file), "wrd"), train_data)
loader:save(string.gsub(train_file, paths.extname(train_file), "wrd.reverse"), train_data, reverse)
-- save index-wise file
train_data = loader:load(train_file, keep)
loader:save(string.gsub(train_file, paths.extname(train_file), "idx"), train_data)
loader:save(string.gsub(train_file, paths.extname(train_file), "idx.reverse"), train_data, reverse)

-- save word-wise file
local valid_data = loader:load(valid_file)
loader:save(string.gsub(valid_file, paths.extname(valid_file), "wrd"), valid_data)
loader:save(string.gsub(valid_file, paths.extname(valid_file), "wrd.reverse"), valid_data, reverse)
-- save index-wise file
valid_data = loader:load(valid_file, keep)
loader:save(string.gsub(valid_file, paths.extname(valid_file), "idx"), valid_data)
loader:save(string.gsub(valid_file, paths.extname(valid_file), "idx.reverse"), valid_data, reverse)

-- save word-wise file
local test_data = loader:load(test_file)
loader:save(string.gsub(test_file, paths.extname(test_file), "wrd"), test_data)
loader:save(string.gsub(test_file, paths.extname(test_file), "wrd.reverse"), test_data, reverse)
-- save index-wise file
test_data = loader:load(test_file, keep)
loader:save(string.gsub(test_file, paths.extname(test_file), "idx"), test_data)
loader:save(string.gsub(test_file, paths.extname(test_file), "idx.reverse"), test_data, reverse)


-- save text file to input dataset file for network input
local reader, pos_info = loader:open(train_file)
local count = 1
while 1 do
  local data = loader:read()
  --if data == nil then break end
  if data == nil then
    local ntime = os.time() + 5
    repeat until os.time() > ntime
    pos_info = loader:view(1, 5) -- reduce the size of the file using veiw()
  end
  print(data)
  if count > 3 then break end
  count = count + 1
end
loader:close()
