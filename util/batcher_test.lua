--[[
Batch Indexer Test

Woohyun Kim(deepcoord@gmail.com)
--]]

require("parser")
require("filereader")
require("wordindexer")
require("inputloader")
require("batcher")
require("tensorbatcher")

local path = require('pl.path')

cmd = torch.CmdLine()
cmd:option('-data_dir','data', 'data directory containing train.txt/valid.txt/test.txt')
cmd:option('-batch_size', 20, 'number of sequences to train on in parallel')
cmd:option('-seq_length', 35, 'number of timesteps to unroll for')
cmd:option('-min_freq', 5, 'minimum frequences for building vocabulary')

opt = cmd:parse(arg)

if not path.exists(opt.data_dir) then
  print("Usage: th batchloader_test.lua -data_dir <data_dir>")
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
--indexer:stats()


-- save text file to input dataset file for network input
local train_batcher = Batcher(indexer, opt.batch_size, opt.seq_length)

-- print out total batches
local reader, pos_info = train_batcher:open(train_file)
train_batcher:make_batches()
train_batcher:print_batches()

-- print out all data, and then print out only five data using just view
count = 1
while 1 do
  local data = train_batcher:read()
  if data == nil then
    local ntime = os.time() + 5
    repeat until os.time() > ntime
    local reduced_pos_info = train_batcher:view(1, 5) -- reduce the size of the file using veiw()
    count = count + 1
    if count > 2 then break end
  end
  print(data)
end

train_batcher:close()


-- peek one more after reading data for batch to support language model with x and y
train_batcher = Batcher(indexer, opt.batch_size, opt.seq_length)
reader, pos_info = train_batcher:open(train_file)
train_batcher:make_batches()

peek = 3 -- peek one more data
for i=1, train_batcher.split_nums do
  print("#splits[" .. i .. "] -------------------------")
  local batches, peeks = train_batcher:next_batch(peek)
  for j=1, #batches do
    io.write("batches[" .. j .. "] = ")
    
    for k=1, #batches[j] do
      if k > 1 then io.write(' ') end
      io.write(train_batcher.indexer.idx2word[batches[j][k]])
    end
    io.write('\n'); io.flush()
  end

  -- print the peeked data
  for j=1, #peeks do
    io.write("##peeks[" .. j .. "] = ")
    
    for k=1, #peeks[j] do
      if k > 1 then io.write(' ') end
      io.write(train_batcher.indexer.idx2word[peeks[j][k]])
    end
    io.write('\n'); io.flush()
  end
end


train_batcher:close()



-- peek one more after reading data for batch to support language model with x and y
local tensor_train_batcher = TensorBatcher(indexer, opt.batch_size, opt.seq_length)
reader, pos_info = tensor_train_batcher:open(train_file)
tensor_train_batcher:make_batches()
tensor_train_batcher:print_batches()


tensor_train_batcher:close()




os.exit()






--[[
-- ready for making batches 
local train_batches = Batcher(indexer, opt.batch_size, opt.seq_length)
local valid_batches = Batcher(indexer, opt.batch_size, opt.seq_length)
local test_batches = Batcher(indexer, opt.batch_size, opt.seq_length)
train_batches:stats()
--valid_batches:stats()
--test_batches:stats()

-- make batches for train
--train_batches:make_batches(train_file)
local train_data = train_batches:load(train_file)
train_batches:split(train_data)

-- make batches for valid
--valid_batches:make_batches(valid_file)
local valid_data = valid_batches:load(valid_file)
valid_batches:split(valid_data)

-- make batches for test
--test_batches:make_batches(test_file)
local test_data = test_batches:load(test_file)
test_batches:replicate(test_data)

print("train batches = " .. #train_batches.batches[1])
print("valid batches = " .. #valid_batches.batches[1])
print("test batches = " .. #test_batches.batches[1])

-- print all the sequences in all the batches
train_batches:print_batches(train_batches.batches)


local batch = train_batches:next_batch()
x = batch.x_batch:float()
y = batch.y_batch:float()
print(x)
print(y)

for i=1, batch.batch_size do
  local x_seq, y_seq, seq_idx = batch:next_seq()
  if seq_idx == 1 then break end
  batch:print_seq(x_seq, y_seq, train_batches.seq_length, train_batches.indexer)
  --x = x_seq:float()
  --print(x)
end

--]]
