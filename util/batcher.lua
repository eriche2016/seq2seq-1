--[[
Batcher 

Woohyun Kim(deepcoord@gmail.com)
--]]

-- inherent from inputloader.lua
local Batcher, parent = torch.class("Batcher", "InputLoader")

-- constructor in lua class
function Batcher:__init(wordindexer, batch_size, seq_length)
  -- call the parent initializer on this child class
  parent.__init(self, wordindexer)

  self.batch_size = batch_size
  self.seq_length = seq_length 

  -- calculated in make_batches()
  -- (format)
  --          |<split 1>|
  -- --------- ----------
  -- batch[1] | x x x x | x x x x | --> <chunk 1>
  -- --------  ----------
  -- batch[2] | x x x x | x x x x | --> <chunk 2>
  -- --------- ----------
  -- batch[.] | . . . . | . . . . |
  -- --------- ----------
  -- batch[k] | x x x x | x x x x | --> <chunk k>
  -- --------- ----------
  self.split_size = 0 -- num of data to process in one batch time
  self.split_nums = 0 -- num of split to process in total batch time
  self.chunk_length = 0 -- total num of data processed by batch

  self.batches = {}
  self.split_cursor = 1
end

function Batcher:stats(verbose)
  self.indexer:stats(verbose)
  print("batch size = " .. self.batch_size)
  print("sequence length = " .. self.seq_length)
  print("split size = " .. self.split_size)
  print("split nums = " .. self.split_nums)
  print("chunk length = " .. self.chunk_length)
  print("# of batches = " .. #self.batches)
  print("split cursor = " .. self.split_cursor)
end

-- make batches
function Batcher:make_batches()
  self.split_size = self.batch_size * self.seq_length
  self.split_nums = math.floor(#self.pos_info / self.split_size)
  if self.split_nums < 1 then
    print(string.format("Input size(%d) is too small. Let the batch_size(%d) or the seq_length(%d) be reduced.", #self.pos_info, self.batch_size, self.seq_length))
    os.exit()
  end

  local pos_info = self:view(1, self.split_size * self.split_nums)
  self.chunk_length = math.floor(#self.pos_info / self.batch_size)

  for i=1, self.batch_size do
    local first = (i-1) * self.chunk_length + 1
    local last = i * self.chunk_length
    self.batches[i] = {first,last}
  end

  self.cursor = 1

  return self.batches
end

-- next batch 
function Batcher:next_batch(peek)
  if self.split_cursor > self.split_nums then self.split_cursor = 1 end

  local batches = {}
  local peeks = {}
  for b=1, self.batch_size do
    local first = self.batches[b][1]
    local last = self.batches[b][2]

    -- read data 
    batches[b] = {}
    self.cursor = (self.split_cursor -1) * self.seq_length + first
    for k=1, self.seq_length do
      if self.cursor > last then break end

      local data = self:read()
      table.insert(batches[b], data)
    end

    -- just peek more data
    if not peek or peek == nil then peek = 0 end
    peeks[b] = {}
    local current = self.cursor
    for k=1, peek do
      local peek_data = self:read()
      if peek_data == nil then peek_data = self:read() end -- when it's the end of file, just read one more
      table.insert(peeks[b], peek_data)
    end
    self.cursor = current
  end 

  self.split_cursor = self.split_cursor + 1

  return batches, peeks
end

function Batcher:print_batches()
  for i=1, self.split_nums do
    print("#splits[" .. i .. "] -------------------------")

    local batches = self:next_batch()
    for j=1, #batches do
      io.write("batches[" .. j .. "] = ")

      for k=1, #batches[j] do
        if k > 1 then io.write(' ') end
        --io.write(batches[j][k])
        io.write(self.indexer.idx2word[batches[j][k]])
      end
      io.write('\n'); io.flush()
    end
  end
end
