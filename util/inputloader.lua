--[[
Input Loader

Woohyun Kim(deepcoord@gmail.com)
--]]
local InputLoader = torch.class("InputLoader")

-- constructor in lua class
function InputLoader:__init(wordindexer)
  self.indexer = wordindexer

  self.reader = nil
  self.pos_info = {}
  self.cursor = 1
end

-- keep means to not convert word to its index
function InputLoader:load(file, keep)
  if keep == nil then keep = true end

  local reader = FileReader(file, self.indexer.tokens)
  local data = {}

  while 1 do
    local words = reader:readwords()
    if words == nil then break end

    -- EOP and EOS have already injected in reader
    for i, word in ipairs(words) do
      if not self.indexer.word2idx[word] then 
        if keep == true then
          data[#data + 1] = self.indexer.tokens.UNK
        else
          data[#data + 1] = self.indexer.word2idx[self.indexer.tokens.UNK]
        end
      else
        if keep == true then
          data[#data + 1] = word
        else 
          data[#data + 1] = self.indexer.word2idx[word]
        end
      end
    end
  end
  reader:close()

  return data
end

-- reverse means to save a pair reversely before EOP
function InputLoader:save(file, data, reverse)
  local writer = torch.DiskFile(file, 'w')
  if not reverse or reverse == false then 
    for i=1, #data do
      if type(data[i]) == "string" then
        if i > 1 then writer:writeString(' ') end
        writer:writeString(data[i])
      elseif type(data[i]) == "number" then
        writer:writeLong(data[i])
      end
    end
  else
    local ahead = true
    local stack = {}
    for i=1, #data do
      if ahead == true then -- eop
        if type(data[i]) == "string" then
          if data[i] == self.indexer.tokens.EOP or data[i] == self.indexer.tokens.EOS then 
            for j=#stack, 1, -1 do
              if j < #stack then writer:writeString(' ') end
              writer:writeString(stack[j])
            end

            if data[i] == self.indexer.tokens.EOP then 
              writer:writeString(' ')
              writer:writeString(data[i])
              ahead = false
            end

            stack = {}
          else
            table.insert(stack, data[i])
          end
        elseif type(data[i]) == "number" then
          if data[i] == self.indexer.word2idx[self.indexer.tokens.EOP] 
             or data[i] == self.indexer.word2idx[self.indexer.tokens.EOS] then 
            for j=#stack, 1, -1 do writer:writeLong(stack[j]) end

            if data[i] == self.indexer.word2idx[self.indexer.tokens.EOP] then 
              writer:writeLong(data[i])
              ahead = false
            end

            stack = {}
          else
            table.insert(stack, data[i])
          end
        end
      else -- eos
        if type(data[i]) == "string" then
          if i > 1 then writer:writeString(' ') end
          writer:writeString(data[i])
          if data[i] == self.indexer.tokens.EOS then ahead = true end
        elseif type(data[i]) == "number" then
          writer:writeLong(data[i])
          if data[i] == self.indexer.word2idx[self.indexer.tokens.EOS] then ahead = true end
        end
      end
    end
  end
  writer:close()
end

-- open it after saving text file to input dataset file for network input
function InputLoader:open(file)
  local reader = FileReader(file, self.indexer.tokens)
  local writer = torch.DiskFile(string.gsub(file, paths.extname(file), "ds"), 'w')

  local pos_info = {}

  while 1 do
    local words = reader:readwords()
    if words == nil then break end

    -- EOP and EOS have already injected in reader
    for i, word in ipairs(words) do
      table.insert(pos_info, writer:position()) -- memorize start position

      if not self.indexer.word2idx[word] then 
        writer:writeLong(self.indexer.word2idx[self.indexer.tokens.UNK])
      else
        writer:writeLong(self.indexer.word2idx[word])
      end
    end
  end
  
  local length = writer:position()
  
  writer:close()
  reader:close()

  -- assign file handler and position information to read data
  if self.reader ~= nil then self:close() end
  self.reader = torch.DiskFile(string.gsub(file, paths.extname(file), "ds"), 'r')
  -- avoid to occur an error in the end of file
  self.reader:quiet()
  self.pos_info = pos_info

  return self.reader, self.pos_info
end

function InputLoader:close()
  if self.reader ~= nil then 
    self.reader:close()
    self.reader = nil
    self.pos_info = {}
    self.cursor = 1
  end
end

-- set a specific range of file
function InputLoader:view(first, last)
  local view_range = {}

  for i=first, last do
    view_range[#view_range + 1] = self.pos_info[i]
  end

  self.pos_info = view_range
  return self.pos_info
end

function InputLoader:read(pos)
  if pos ~= nil and type(pos) == "number" then 
    self.reader:seek(pos) 
  elseif self.cursor > #self.pos_info then 
    self.reader:seekEnd()
  else
    self.reader:seek(self.pos_info[self.cursor])
    self.cursor = self.cursor + 1
  end

  local data = self.reader:readLong()
  local next_pos = self.reader:position()

  -- it have met the end of file
  if self.reader:hasError() == true then 
    -- it might be continuted to read from the starting position
    self.reader:clearError()

    self.cursor = 1

    data = nil
    next_pos = self.reader:position()
  end

  return data, next_pos
end
