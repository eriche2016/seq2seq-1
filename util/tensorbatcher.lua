--[[
TensorBatcher 

Woohyun Kim(deepcoord@gmail.com)
--]]

-- inherent from batcher.lua
local TensorBatcher, parent = torch.class("TensorBatcher", "Batcher")

-- constructor in lua class
function TensorBatcher:__init(wordindexer, batch_size, seq_length)
  -- call the parent initializer on this child class
  parent.__init(self, wordindexer, batch_size, seq_length)
end

-- overriden
-- next batch to tensor
function TensorBatcher:next_batch()
  if self.split_cursor > self.split_nums then self.split_cursor = 1 end

  local x_batches = torch.LongTensor(self.batch_size, self.seq_length)
  local y_batches = torch.LongTensor(self.batch_size, self.seq_length)
  local x_char_batches = torch.LongTensor(self.batch_size, self.seq_length, self.indexer.max_word_l+2)

  for b=1, self.batch_size do
    local first = self.batches[b][1]
    local last = self.batches[b][2]

    -- read data 
    self.cursor = (self.split_cursor -1) * self.seq_length + first
    for k=1, self.seq_length do
      if self.cursor > last then break end

      local data = self:read()
      x_batches[b][k] = data

      -- char
      local word = self.indexer.idx2word[data]
      local chars = self.indexer:word2chars(word)
      -- zero-padding
      x_char_batches[b][k]:fill(1)
      -- add start of word
      x_char_batches[b][k][1] = self.indexer.char2idx[self.indexer.tokens.START]
      local limit = math.min(#chars, self.indexer.max_word_l)
      for c=1, limit do
        if c >= self.indexer.max_word_l then break end
        x_char_batches[b][k][c+1] = self.indexer.char2idx[chars[c]]
      end
      -- add end of word
      x_char_batches[b][k][limit + 2] = self.indexer.char2idx[self.indexer.tokens.END]
    end

    -- shift copy
    y_batches[b] = x_batches[b]:clone()
    y_batches[b]:sub(1,-2):copy(x_batches[b]:sub(2,-1))

    -- the data of y is the first of the next batch
    local current = self.cursor
    local peek_data = self:read()
    if peek_data == nil then peek_data = self:read() end -- when it's the end of file, just read one more
    y_batches[b][-1] = peek_data
    self.cursor = current
  end 

  self.split_cursor = self.split_cursor + 1

  return x_batches, y_batches, x_char_batches
end

-- overridden
-- print out next batch
function TensorBatcher:print_batches()
  for i=1, self.split_nums do
    print("#splits[" .. i .. "] -------------------------")

    local x_batches, y_batches, x_char_batches = self:next_batch()
    assert(x_batches:size(1) == y_batches:size(1))
    assert(x_batches:size(2) == y_batches:size(2))
    for j=1, x_batches:size(1) do
      io.write("x[" .. j .. "] = ")
      for k=1, x_batches:size(2) do
        if k > 1 then io.write(' ') end
        io.write(self.indexer.idx2word[x_batches[j][k]])
      end
      io.write('\n'); io.flush()

      io.write("y[" .. j .. "] = ")
      for k=1, y_batches:size(2) do
        if k > 1 then io.write(' ') end
        io.write(self.indexer.idx2word[y_batches[j][k]])
      end
      io.write('\n'); io.flush()

      io.write("c[" .. j .. "] = ")
      for k=1, x_char_batches:size(2) do
        if k > 1 then io.write(' ') end
        for l=1, x_char_batches:size(3) do
          io.write(self.indexer.idx2char[x_char_batches[j][k][l]])
          if x_char_batches[j][k][l] == self.indexer.char2idx[self.indexer.tokens.END] then break end
        end
      end
      io.write('\n'); io.flush()
    end
  end
end
