--[[
Encoder

Woohyun Kim(deepcoord@gmail.com)
--]]

local Encoder = torch.class("Encoder")

function Encoder:__init(indexer, network)
  self.opt = network.opt
  self.indexer = indexer
  self.network = network 
end

function Encoder:split(text, sep)
  if sep ~= nil then self.sep = sep end

  local t = {}
  local i = 1
  for str in string.gmatch(text, "([^"..self.sep.."]+)") do
    t[i] = str; i = i + 1
  end

  return t
end

-- parse
function Encoder:parse(line, tokens)
  if line == nil then return nil end

  -- contain puntuations after sperating from word
  line = string.gsub(line, "([%p])([%w]+)", "%1 %2")
  line = string.gsub(line, "([%w]+)([%p])", "%1 %2")

  -- use the given word separater
  if tokens.SEP == nil then tokens.SEP = "%s" end

  local words = self:split(line, tokens.SEP)
  for i=1, #words do words[i] = string.lower(words[i]) end
  --for i=1, #words do words[i] = words[i] end
  return words
end

-- parse text with word indexes
function Encoder:parse2(text)
  local wlist = {}
  local words = parent.tokenize(self, text, self.opt.tokens)
  for i=1, #words do table.insert(wlist, words[i]) end

  if self.opt.tokens.EOP ~= nil then table.insert(wlist, self.opt.tokens.EOP) end

  local x = torch.LongTensor(#wlist)
  local x_char = torch.LongTensor(#wlist, self.indexer.max_word_l+2)

  for i, w in ipairs(wlist) do
    --x[i] = self.indexer.word2idx[w]
    x[i] = self.indexer.word2idx[w] ~= nil and self.indexer.word2idx[w] or self.indexer.word2idx[self.opt.tokens.UNK]
    -- chars
    local chars = self.indexer:word2chars(w)
    local limit = math.min(#chars, self.indexer.max_word_l)
    -- zero-padding
    x_char[i]:fill(1)
    -- add start of word
    x_char[i][1] = self.indexer.char2idx[self.indexer.tokens.START]
    for c=1, limit do
      if c >= self.indexer.max_word_l then break end
      x_char[i][c+1] = self.indexer.char2idx[chars[c]]
    end
    -- add end of word
    x_char[i][limit+2] = self.indexer.char2idx[self.indexer.tokens.END]
  end

  return x, x_char
end

-- for easy switch between using words/chars (or both)
function Encoder:get_input(x, x_char, prev_states)
  local u = {}
  if self.opt.use_chars == 1 then table.insert(u, x_char[{}]) end
  if self.opt.use_words == 1 then table.insert(u, x[{}]) end
  for i = 1, #prev_states do table.insert(u, prev_states[i]) end
  return u
end

-- extract n-best
function Encoder:nbest(output, n, skips)
    if not n then n = 5 end
    local sorted, oidx = output:sort(2, true) -- descending
    --for i=1, n do print(i .. "\t" .. oidx[1][i] .. " " .. idx2word[oidx[1][i]]) end

    if not skips then
        return oidx[1]:narrow(1,1,n)
    else
        local k = 1
        local eos = false
        local harvest = torch.LongTensor(n):fill(0)
        for i=1, oidx:size(2) do
          if k > n then break end

          local skipped = false
          for _, skip in ipairs(skips) do
              if i==1 and skip == oidx[1][i] and skip == self.indexer.word2idx[self.opt.tokens.EOS] then eos = true end
              if i==1 and skip == oidx[1][i] and skip == self.indexer.word2idx[self.opt.tokens.EOP] then eos = true end
              if skip == oidx[1][i] then skipped = true end
          end
          if skipped == false then
              harvest[k] = oidx[1][i]
              k = k + 1
          end
          --if eos == true then break end
        end

        return harvest, eos
    end
end



-- encode text in fixed-length vector representation
-- and return predicted output by softmax
function Encoder:encode(text)
  if self.opt.hsm > 0 then
    self.network.criterion:change_bias()
  end

  local init_state = self.network:get_init_state(2)
  local rnn_state = {[0] = init_state}

  local wlist = self:parse(text, self.opt.tokens)
  -- add <eop> for sequence to sequence model
  --table.insert(wlist, "<eop>")

  local xt = torch.LongTensor(#wlist+1)

  if not self.opt.reverse or self.opt.reverse == 0 then
    for i, w in ipairs(wlist) do
      xt[i] = self.indexer.word2idx[w] ~= nil and self.indexer.word2idx[w] or self.indexer.word2idx[self.opt.tokens.UNK]
    end
  elseif self.opt.reverse == 1 then
    for i=#wlist, 1, -1 do
      local w = wlist[i]
      xt[(#wlist+1)-i] = self.indexer.word2idx[w] ~= nil and self.indexer.word2idx[w] or self.indexer.word2idx[self.opt.tokens.UNK]
    end
  end

  -- add <eop> for sequence to sequence model
  if self.indexer.word2idx["<eop>"] ~= nil then
    xt[xt:size(1)] = self.indexer.word2idx["<eop>"]
  else
    xt = xt:sub(1, xt:size(1)-1)
  end

  for t = 1, xt:size(1) do
    local chars = self.indexer:word2chars(self.indexer.idx2word[xt[t]])
    local x_char = torch.LongTensor(2, self.indexer.max_word_l)
    x_char[1][1] = self.indexer.char2idx[self.opt.tokens.START]
    x_char[2][1] = self.indexer.char2idx[self.opt.tokens.START]
    for c=1, #chars do
      x_char[1][c+1] = self.indexer.char2idx[chars[c]] ~= nil and self.indexer.char2idx[chars[c]] or 1
      x_char[2][c+1] = self.indexer.char2idx[chars[c]] ~= nil and self.indexer.char2idx[chars[c]] or 1
    end
    x_char[1][#chars+2] = self.indexer.char2idx[self.opt.tokens.END]
    x_char[2][#chars+2] = self.indexer.char2idx[self.opt.tokens.END]
    for c=#chars+3, self.indexer.max_word_l do
      x_char[1][c] = 1
      x_char[2][c] = 1
    end

    local x = torch.LongTensor(2)
    x[1] = xt[t]
    x[2] = xt[t]

    if self.opt.gpuid >= 0 then
      x = x:float():cuda()
      x_char = x_char:float():cuda()
    end

    local lst = self.network.rnn:forward(self:get_input(x, x_char, rnn_state[t-1]))
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]:clone()) end
    prediction = lst[#lst]:clone()

    -- print out n-best
    --local best = self:nbest(prediction)
    --print(self.indexer.idx2word[xt[t]] ~= nil and self.indexer.idx2word[xt[t]] or xt[t])
    --for i=1, best:size(1) do print(i .. "\t" .. best[i] .. " " .. self.indexer.idx2word[best[i]]) end
  end

  return rnn_state[#rnn_state], prediction
end
