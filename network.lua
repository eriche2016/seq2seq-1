--[[
Network
-- create network layers and architectures

Woohyun Kim(deepcoord@gmail.com)
--]]

local Network = torch.class("Network")

-- model utility
local model_utils = require('util.model_utils')

-- get layers which will be referenced layer (during SGD or introspection)
local word_vecs = nil
local char_vecs = nil
local cnn = nil
function get_layer(layer)
  local tn = torch.typename(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs' then
      word_vecs = layer
    elseif layer.name == 'char_vecs' then
      char_vecs = layer
    elseif layer.name == 'cnn' then
      cnn = layer
    end
  end
end

-- constructor in lua class
function Network:__init(opt)
  self.opt = opt
  self.rnn = {}
  --self.criterion -- initialized later
  --self.params  -- initialized later
  --self.grad_params  -- initialized later
  self.init_state = {}
  self.init_state_global = {}

  -- references for layers
  self.layer = {}
  self.layer.word_vecs = nil
  self.layer.char_vecs = nil 
  self.layer.cnn = nil 
end

-- takes a list of tensors and returns a list of cloned tensors
function Network:clone_list(tensor_list, zero_too)
  local out = {}
  for k,v in pairs(tensor_list) do
    out[k] = v:clone()
    if zero_too then out[k]:zero() end
  end
  return out
end

-- get initial state based on batch_size
function Network:get_init_state(batch_size)
  if not batch_size then batch_size = self.opt.batch_size end

  -- set the initial state of the cell/hidden states
  local init_state = {}
  for L=1,self.opt.num_layers do
    local h_init = torch.zeros(batch_size, self.opt.rnn_size)
    if self.opt.gpuid >=0 and self.opt.opencl == 0 then h_init = h_init:cuda() end
    if self.opt.gpuid >=0 and self.opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone()) -- hidden state
    if self.opt.model == 'lstm' or self.opt.model == 'lstmtdnn' or self.opt.model == 'highwaymlp' then
      table.insert(init_state, h_init:clone()) -- cell state
    end
  end

  return self:clone_list(init_state)
end

-- set model
function Network:make_model(indexer)
  self.indexer = indexer

  -- define the model: prototypes for one timestep, then clone them in time
  print('creating an ' .. self.opt.model .. ' with ' .. self.opt.num_layers .. ' layers and ' .. #self.indexer.idx2word .. ' vocab size (' .. #self.indexer.idx2char .. ' characters)')

  -- set up training model
  if self.opt.model == 'lstmtdnn' then
    -- NOTE: LSTM-TDNN contains a softmax layer in the last layer
    local LSTMTDNN = LSTMTDNN()
    self.rnn = LSTMTDNN:lstmtdnn(self.opt.rnn_size, self.opt.num_layers, self.opt.dropout, #self.indexer.idx2word,
                  self.opt.word_vec_size, #self.indexer.idx2char, self.opt.char_vec_size, self.opt.feature_maps, 
                  self.opt.kernels, self.indexer.max_word_l, self.opt.use_words, self.opt.use_chars, 
                  self.opt.batch_norm, self.opt.highway_layers, self.opt.hsm)

  elseif self.opt.model == 'tdnn' then
    print("not yet"); os.exit()
    local TDNN = TDNN()
  elseif self.opt.model == 'highwaymlp' then
    print("not yet"); os.exit()
    local HighwayMLP = HighwayMLP()
  elseif self.opt.model == 'lstm' then
    print("not yet"); os.exit()
    local LSTM = LSTM()
    self.rnn = LSTM:lstm(self.vocab_size, self.opt.rnn_size, self.opt.num_layers, self.opt.dropout)
  elseif self.opt.model == 'gru' then
    print("not yet"); os.exit()
    local RNN = RNN()
    self.rnn = GRU:gru(self.vocab_size, self.opt.rnn_size, self.opt.num_layers, self.opt.dropout)
  elseif self.opt.model == 'rnn' then
    print("not yet"); os.exit()
    local GRU = GRU()
    self.rnn = RNN:rnn(self.vocab_size, self.opt.rnn_size, self.opt.num_layers, self.opt.dropout)
  end

  -- set up criterion
  if self.opt.hsm == -1 then self.opt.hsm = torch.round(torch.sqrt(#self.indexer.idx2word)) end
  if self.opt.hsm > 0 then
    -- partition into self.opt.hsm clusters
    -- we want roughly equal number of words in each cluster
    self.mapping = torch.LongTensor(#self.indexer.idx2word, 2):zero()
    local n_in_each_cluster = #self.indexer.idx2word / self.opt.hsm
    local _, idx = torch.sort(torch.randn(#self.indexer.idx2word), 1, true)
    local n_in_cluster = {} --number of tokens in each cluster
    local c = 1
    for i = 1, idx:size(1) do
      local word_idx = idx[i]
      if n_in_cluster[c] == nil then
        n_in_cluster[c] = 1
      else
        n_in_cluster[c] = n_in_cluster[c] + 1
      end
      self.mapping[word_idx][1] = c
      self.mapping[word_idx][2] = n_in_cluster[c]
      if n_in_cluster[c] >= n_in_each_cluster then
        c = c+1
      end
      if c > self.opt.hsm then --take care of some corner cases
        c = self.opt.hsm
      end
    end
    print(string.format('using hierarchical softmax with %d classes', self.opt.hsm))

    self.criterion = nn.HLogSoftMax(self.mapping, self.opt.rnn_size)
  else
    -- training criterion (negative log likelihood)
    self.criterion = nn.ClassNLLCriterion()
  end

  -- set the initial state of the cell/hidden states
  self.init_state = {}
  for L=1,self.opt.num_layers do
    local h_init = torch.zeros(self.opt.batch_size, self.opt.rnn_size)
    if self.opt.gpuid >=0 and self.opt.opencl == 0 then h_init = h_init:cuda() end
    if self.opt.gpuid >=0 and self.opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(self.init_state, h_init:clone()) -- hidden state
    if self.opt.model == 'lstm' or self.opt.model == 'lstmtdnn' or self.opt.model == 'highwaymlp' then
      table.insert(self.init_state, h_init:clone()) -- cell state
    end
  end

  -- ready for the global initial state from the initial state
  self.init_state_global = self:clone_list(self.init_state)

  -- ship the model to the GPU if desired
  if self.opt.gpuid >= 0 and self.opt.opencl == 0 then
    self.rnn:cuda()
    self.criterion:cuda()
  end
  if self.opt.gpuid >= 0 and self.opt.opencl == 1 then
    self.rnn:cl()
    self.criterion:cl()
  end
end

-- initialize network parameters, and network layers
function Network:clone_model()
  -- put the network things into one flattened parameters tensor
  local params, grad_params = model_utils.combine_all_parameters(self.rnn)
  self.params = params
  self.grad_params = grad_params

  -- hsm has its own params
  if self.opt.hsm > 0 then
    local hsm_params, hsm_grad_params = self.criterion:getParameters()
    self.hsm_params = hsm_params
    self.hsm_grad_params = hsm_grad_params
    self.hsm_params:uniform(-self.opt.param_init, self.opt.param_init)

    print('number of parameters in the model: ' .. self.params:nElement() + self.hsm_params:nElement())
  else
    print('number of parameters in the model: ' .. self.params:nElement())
  end

  -- initialization
  self.params:uniform(-self.opt.param_init, self.opt.param_init) -- small numbers uniform if starting from scratch

  -- get the referred layers
  self.rnn:apply(get_layer)
  self.layer.word_vecs = word_vecs
  self.layer.char_vecs = char_vecs
  self.layer.cnn = cnn

  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if self.opt.model == 'lstm' or self.opt.model == 'lstmtdnn' or self.opt.model == 'highwaymlp' then
    for layer_idx = 1, self.opt.num_layers do
      for _,node in ipairs(self.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i, f, o, g, so f is the 2nd block of weights
          node.data.module.bias[{{self.opt.rnn_size+1, 2*self.opt.rnn_size}}]:fill(1.0)
        end
      end
    end
  end

  -- cloning modules in a layer as much as time-steps
  -- modules contains rnn, and criterion, but not softmax here(softmax can be found in LSTMTDNN)
  self.clones = {}
  print('cloning rnn model(' .. self.opt.model .. ') as much as time steps(' .. self.opt.seq_length .. ')')
  self.clones["rnn"] = model_utils.clone_many_times(self.rnn, self.opt.seq_length, not self.rnn.parameters)
  print('cloning criterion as much as time steps(' .. self.opt.seq_length .. ')')
  self.clones["criterion"] = model_utils.clone_many_times(self.criterion, self.opt.seq_length, not self.criterion.parameters)
end
