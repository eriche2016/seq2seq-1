--[[
Decoder

Woohyun Kim(deepcoord@gmail.com)
--]]

-- inherent from parser.lua
local Decoder = torch.class("Decoder")

function Decoder:__init(indexer, network)
  self.opt = network.opt
  self.indexer = indexer
  self.network = network 
end

-- for easy switch between using words/chars (or both)
function Decoder:get_input(x, x_char, prev_states)
  local u = {}
  if self.opt.use_chars == 1 then table.insert(u, x_char[{}]) end
  if self.opt.use_words == 1 then table.insert(u, x[{}]) end
  for i = 1, #prev_states do table.insert(u, prev_states[i]) end
  return u
end

function Decoder:split(text, sep)
  if sep ~= nil then self.sep = sep end

  local t = {}
  local i = 1
  for str in string.gmatch(text, "([^"..self.sep.."]+)") do
    t[i] = str; i = i + 1
  end

  return t
end

function Decoder:parse(line, tokens)
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
function Decoder:parse2(text)
  local wlist = {}
  local words = parent.tokenize(self, text, self.opt.tokens)
  for i=1, #words do table.insert(wlist, words[i]) end

  if self.opt.tokens.EOP ~= nil then table.insert(wlist, self.opt.tokens.EOP) end

  local x = torch.LongTensor(#wlist)
  local x_char = torch.LongTensor(#wlist, self.indexer.max_word_l+2)

  for i, w in ipairs(wlist) do
    x[i] = self.indexer.word2idx[w] ~= nil and self.indexer.word2idx[w] or self.indexer.word2idx[self.opt.tokens.UNK]
    -- chars
    local chars = self.indexer:word2chars(w)
    local limit = math.min(#chars, self.indexer.max_word_l)
    -- zero-padding
    x_char[i]:fill(1)
    for c=1, limt do
      if c >= self.indexer.max_word_l then break end
      x_char[i][c+1] = self.indexer.char2idx[chars[c]]
    end
    -- add end of word
    x_char[i][limit+2] = self.indexer.char2idx[self.indexer.tokens.END]
  end

  return x, x_char
end

-- extract n-best
function Decoder:nbest(output, n, skips)
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

-- decode a fixed-length vector representation to a generated sequence
function Decoder:decode(context, output, length)
  local sequence = {}
  if not length then length = 20 end

  -- make the previous state at time 1 to be zeros
  local init_state = self.network:get_init_state(2)
  local rnn_state = {[0] = self.network:clone_list(context)}
  local prediction = output

  local skips = {}
  table.insert(skips, self.indexer.word2idx["-"])
  table.insert(skips, self.indexer.word2idx[self.opt.tokens.UNK])
  table.insert(skips, self.indexer.word2idx[self.opt.tokens.EOP])
  table.insert(skips, self.indexer.word2idx[self.opt.tokens.EOS])
  local selected = 1 -- argmaxing
  for t = 1, length do
    local best, eos = self:nbest(prediction, 5, skips)
    --print("---" .. t)
    --for i=1, best:size(1) do
    --  if best[i] ~= 0 then print(i .. "\t" .. best[i] .. " " .. self.indexer.idx2word[best[i]]) end
    --end

    if eos == true then break end
    --if eos == true and best[1] == word2idx[opt.tokens.EOS] then break end

    local chars = self.indexer:word2chars(self.indexer.idx2word[best[selected]])
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
    x[1] = best[selected]
    x[2] = best[selected]
    local w = self.indexer.idx2word[x[1]] ~= nil and self.indexer.idx2word[x[1]] or nil
    if w == nil then break end

    table.insert(sequence, w)

    local lst = self.network.rnn:forward(self:get_input(x, x_char, rnn_state[t-1]))
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]:clone()) end
    prediction = lst[#lst] :clone()
  end

  return rnn_state[#rnn_state], sequence
end


-- decode a fixed-length vector representation to a generated sequence
function Decoder:decode2(context, output, length)
  local sequence = {}
  if not length then length = 20 end

  -- make the previous state at time 1 to be zeros
  local rnn_state = {[0] = self.network:clone_list(context)}
  local prediction = output:clone()

  -- comment out to make variations for the same sequences
  --self.network.rnn:evaluate() 
  
  -- start sampling / argmaxing
  local selected = 1
  for t=1, length do
    local nbest, eos = self:nbest(prediction)
    if eos == true and nbest[1] == self.indexer.word2idx[self.opt.tokens.EOS] then -- if it was too short, what would we do?
      break
    end

    for i=1, nbest:size(1) do 
      --io.write(self.indexer.idx2word[nbest[i]] .. "(" .. nbest[i] .. "), ")
    end
    --io.write('\n'); io.flush()

    x = torch.LongTensor(1)
    x[1] = nbest[selected]

    local w = self.indexer.idx2word[x[1]] ~= nil and self.indexer.idx2word[x[1]] or nil
    if w == nil then 
      break 
    end

    table.insert(sequence, w)

    -- forward the rnn for next word
    local fwd_state = self.network.rnn:forward(self:get_input(x, rnn_state[t-1]))
    rnn_state[t] = {}
    for i=1,#self.init_state do table.insert(rnn_state[t], fwd_state[i]:clone()) end

    prediction = fwd_state[#fwd_state]:clone()
  end

  return rnn_state[#rnn_state], sequence
end
