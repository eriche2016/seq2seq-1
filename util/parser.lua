--[[
Parser

Woohyun Kim(deepcoord@gmail.com)
--]]

local Parser = torch.class("Parser")

-- constructor in lua class
function Parser:__init()
  self.sep = "%p%s"
  self.esc = "%p%s"
  self.tokens = {}
  self.tokens.SEP = "%s"
end

-- split on separator
function Parser:split(text, sep)
  if sep ~= nil then self.sep = sep end

  local t = {}
  local i = 1
  for str in string.gmatch(text, "([^"..self.sep.."]+)") do
    t[i] = str; i = i + 1
  end

  return t
end

-- trim
function Parser:trim(data, esc)
  if esc ~= nil then self.esc = esc end
  
  local t = string.gsub(data, "["..self.esc.."]", "")

  return t
end

-- copy table recursively
function Parser:deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- compare two tables
function Parser:comparetables(t1, t2)
  if #t1 ~= #t2 then return false end
  for i=1,#t1 do
    if t1[i] ~= t2[i] then return false end
  end
  return true
end

-- subrange table
function Parser:subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

-- tokenize
function Parser:tokenize(text, tokens)
  if text == nil then return nil end
  -- use the given word separater
  if tokens ~= nil then self.tokens = tokens end

  -- contain puntuations after sperating from word
  text = string.gsub(text, "([%p])([%w]+)", "%1 %2")
  text = string.gsub(text, "([%w]+)([%p])", "%1 %2")

  local words = self:split(text, tokens.SEP)
  for i=1, #words do words[i] = string.lower(words[i]) end
  return words
end

