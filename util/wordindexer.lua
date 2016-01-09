--[[
Word Indexer

Woohyun Kim(deepcoord@gmail.com)
--]]

local WordIndexer = torch.class("WordIndexer")

-- check out if "luautf8" was installed by luarocks
local utf8_ok, utf8 = pcall(require, 'lua-utf8')
if not utf8_ok then utf8 = string end -- only for utf8.len

-- constructor in lua class
function WordIndexer:__init(tokens, freq)
  -- tokens
  self.tokens = {}

  if not tokens then
    self.tokens.SEP = "%s" -- word separator
    self.tokens.EOP = "<eop>" -- end of a pair
    self.tokens.EOS = "<eos>" -- end of a sentence
    self.tokens.UNK = "<unk>" -- unkown word
    -- char
    self.tokens.ZEROPAD = ' ' -- zero-pad
    self.tokens.START = '{' -- start of a word
    self.tokens.END = '}' -- end of a word
  else
    self.tokens = tokens
  end

  -- vocab
  self.freqs = {}
  self.minimum_freq = freq ~= nil and freq or 1

  -- word
  self.idx2word = { self.tokens.UNK, self.tokens.EOP, self.tokens.EOS }
  self.word2idx = {} 
  self.word2idx[self.tokens.UNK] = 1
  self.word2idx[self.tokens.EOP] = 2
  self.word2idx[self.tokens.EOS] = 3

  -- char
  self.max_word = nil
  self.max_word_l = 0 -- max word length of the corpus
  self.idx2char = { self.tokens.ZEROPAD, self.tokens.START, self.tokens.END }
  self.char2idx = {}
  self.char2idx[self.tokens.ZEROPAD] = 1
  self.char2idx[self.tokens.START] = 2
  self.char2idx[self.tokens.END] = 3
end

-- char
function WordIndexer:word2chars(word)
  local chars = {}

  if not utf8_ok then -- not utf8
    for char in string.gmatch(word, ".") do chars[#chars + 1] = char end
  else -- utf8
    for _, char in utf8.next, word do
      chars[#chars + 1] = utf8.char(char) -- conver code to character
    end
  end

  return chars
end

function WordIndexer:reuse(idx2word, word2idx, idx2char, char2idx, max_word_l)
  self.idx2word = idx2word
  self.word2idx = word2idx
  self.idx2char = idx2char
  self.char2idx = char2idx

  self.max_word_l = max_word_l
end

-- add vocabularies after loading file to build up vocabularies
function WordIndexer:add(file)
  local reader = FileReader(file, self.tokens) 

  -- read words by the line from file
  while 1 do
    local words = reader:readwords()
    if words == nil then break end

    -- calculate word occurances
    for i, word in ipairs(words) do
      if not self.freqs[word] then
        self.freqs[word] = 1
        self.max_word = self.max_word_l < utf8.len(word) and word or self.max_word
        -- char
        self.max_word_l = math.max(self.max_word_l, utf8.len(word))
      else
        self.freqs[word] = self.freqs[word] + 1
      end
    end
  end
  reader:close()

  -- limit the vocaularies to the given minimum frequences
  for word, wc in pairs(self.freqs) do
    -- remove the word which is smaller than minimum_freq
    if wc > self.minimum_freq then
      if not self.word2idx[word] then
        self.idx2word[#self.idx2word + 1] = word
        self.word2idx[word] = #self.idx2word
      end

      -- char
      local chars = self:word2chars(word)
      for i=1, #chars do
        char = chars[i]
        if self.char2idx[char] == nil then
          self.idx2char[#self.idx2char + 1] = char
          self.char2idx[char] = #self.idx2char
        end
      end

    end
  end
end

function WordIndexer:save(vocab_file)
  print('saving' .. vocab_file)
  torch.save(vocab_file, { self.idx2word, self.word2idx, self.idx2char, self.char2idx })
end

function WordIndexer:stats(verbose)
  if verbose ~= nil and verbose == true then
    for idx, word in ipairs(self.idx2word) do
      print(string.format("[%d]%s (%d)", idx, word, self.freqs[word] ~= nil and self.freqs[word] or 0))
    end
    for idx, char in ipairs(self.idx2char) do
      print(string.format("[%d]%s", idx, char))
    end
  end

  print(string.format("idx2word size = %d", #self.idx2word))
  print(string.format("maximum word length = %d (%s)", self.max_word_l, self.max_word))
end

