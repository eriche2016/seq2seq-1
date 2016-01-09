--[[
File Reader inherent from Parser

Woohyun Kim(deepcoord@gmail.com)
--]]

-- inherent from parser.lua
local FileReader, parent = torch.class("FileReader", "Parser")

-- constructor in lua class
function FileReader:__init(file, tokens)
  -- call the parent initializer on this child class
  parent.__init(self)
  
  if tokens ~= nil then self.tokens = tokens end

  self.file = file
  self.fd = nil
end

function FileReader:close()
  if self.fd ~= nil then
    -- end of file
    self.fd:close()
    self.fd = nil
  end
end

-- read line from file
function FileReader:readline()
  if self.fd == nil then
    self.fd = io.open(self.file, "r")
  end

  local line = self.fd:lines() -- return iterator
  if line ~= nil then 
    return line()
  else
    self:close()
    return nil
  end
end

-- read words by the line from file
function FileReader:readwords(tokens)
  local line = self:readline()
  local words = self:tokenize(line, tokens)
  return words
end

-- overriden from Parser:tokenize()
-- a line is assumed as input to automatically add EOS
-- <tab> in the line will be replaced to EOP given tokens.EOP
function FileReader:tokenize(line, tokens)
  if line == nil then return nil end
  if tokens ~= nil then self.tokens = tokens end

  -- keep pre-assigned tokens such as <eop>, <eos>, <unk> from puntuations
  if self.tokens.EOP ~= nil then line = string.gsub(line, self.tokens.EOP, "\t") end
  if self.tokens.EOS ~= nil then line = string.gsub(line, self.tokens.EOS, "\t") end
  if self.tokens.UNK~= nil then line = string.gsub(line, self.tokens.UNK, "\t") end

  -- contain puntuations after sperating from word
  line = string.gsub(line, "([%p])([%w]+)", "%1 %2")
  line = string.gsub(line, "([%w]+)([%p])", "%1 %2")

  -- add EOP and EOS
  if self.tokens.EOP ~= nil then line = string.gsub(line, "\t", " " .. self.tokens.EOP.. " ") end
  if self.tokens.EOS ~= nil then line = line .. " " .. self.tokens.EOS end

  -- use the given word separater
  if self.tokens.SEP == nil then self.tokens.SEP = "%s" end

  -- remove the reserved tokens such as start('{') and end('}') of a word
  if self.tokens.START ~= nil then string.gsub(line, self.tokens.START, '') end
  if self.tokens.END ~= nil then string.gsub(line, self.tokens.END, '') end

  local words = self:split(line, self.tokens.SEP)
  for i=1, #words do words[i] = string.lower(words[i]) end
  return words
end
