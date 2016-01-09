--[[
Optimizer

Woohyun Kim(deepcoord@gmail.com)
--]]

local Optimizer, parent = torch.class("Optimizer", "Parser")

function Optimizer:__init(indexer, network)
  -- call the parent initializer on this child class
  parent.__init(self)

  self.opt = network.opt
  self.indexer = indexer
  self.network = network 
  self.rnn_state = {}
end

-- input with the previous cell/hidden
function Optimizer:get_input(x, x_char, t, prev_states)
  local u = {}
  -- chars input at time t
  if self.opt.use_chars == 1 then table.insert(u, x_char[{{},t}]) end
  -- word input at time t
  if self.opt.use_words == 1 then table.insert(u, x[{{},t}]) end
  -- cell/hidden state at time (t-1)
  for i = 1, #prev_states do table.insert(u, prev_states[i]) end
  return u
end

-- rank n-best
function Optimizer:nbest(output, n)
  if not n then n = 5 end
  local sorted, oidx = output:sort(2, true) -- descending
  -- for i=1, n do print(i .. "\t" .. oidx[1][i] .. " " .. self.indexer.idx2word[oidx[1][i]]) end
  
  return oidx[1]:narrow(1,1,n)
end

-- initialize gradients before forward propagation
function Optimizer:initialize_grad_parameters()
  self.network.grad_params:zero()
end

-- forward pass
function Optimizer:forward(x, y, x_char, verbose)
  -- make the previous state at time 1 to be zeros
  self.rnn_state = {[0] = self.network.init_state_global}

  local predictions = {}  -- softmax outputs
  local loss = 0

  -- forward, and reserve the states as much as there are timesteps
  local t = 1
  for t=1,self.opt.seq_length do -- forward to seq_length timesteps
    self.network.clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    local fwd_state = self.network.clones.rnn[t]:forward(self:get_input(x, x_char, t, self.rnn_state[t-1]))
    self.rnn_state[t] = {}

    local i = 1
    -- reserve the current states after forwarding (without output)
    for i=1,#self.network.init_state do table.insert(self.rnn_state[t], fwd_state[i]) end
    -- reserve the last state as output for softmax predictions
    predictions[t] = fwd_state[#fwd_state]
    -- calculate the loss between pred and y in softmax
    if y ~= nil then loss = loss + self.network.clones.criterion[t]:forward(predictions[t], y[{{}, t}]) end

    if verbose ~= nil and verbose == true then
      local nbest = self:nbest(predictions[t])
      print(self.indexer.idx2word[x[t]])
      for i=1, nbest:size(1) do print(i .. "\t" .. nbest[i] .. " " .. self.indexer.idx2word[nbest[i]]) end
    end
  end
  -- average of the losses in timesteps
  loss = loss / self.opt.seq_length

  return self.rnn_state[#self.rnn_state], predictions, torch.exp(loss)
end

-- backward pass 
function Optimizer:backward(x, y, x_char, predictions)
  -- the backward state(grad_input) at time t is supposed to be zeros
  local gradients = {[self.opt.seq_length] = self.network:clone_list(self.network.init_state, true)}
  local t = self.opt.seq_length
  for t=self.opt.seq_length,1,-1 do -- backward from seq_length timesteps
    -- grad_output: gradient descent from output(y[t]) to prediction
    local grad_output = self.network.clones.criterion[t]:backward(predictions[t], y[{{}, t}])
    table.insert(gradients[t], grad_output)
    -- reserve a gradient from pred to y, and a gradient from next_state
    -- in fact, the caluated gradients[t] will be added into the end of the self.rnn_state[t-1]
    table.insert(self.rnn_state[t-1], gradients[t])

    -- grad_input: gradient descent from cell/hidden to input(x[t])
    local grad_input = self.network.clones.rnn[t]:backward(self:get_input(x, x_char, t, self.rnn_state[t-1]), gradients[t])

    -- the backward state(grad_input) at time (t-1) is grad_input at time t
    gradients[t-1] = {}
    local tmp = self.opt.use_words + self.opt.use_chars -- not the safest way but quick
    local k; local v
    for k,v in pairs(grad_input) do
      -- grad_input will be the same structure as the input of forward()
      -- (e.g.) x[t] and rnn_state[t-1]
      if k > tmp then -- reserve the gradients on cell/hidden after ignoring gradient on x[t]
        gradients[t-1][k-tmp] = v
      end
    end
  end

  -- in fact, the caluated gradients[t] will be added into the end of the self.rnn_state[t-1]
  -- so the returned gradients don't need to be used
  return gradients
end

-- update parameters
function Optimizer:update_parameters(lr, train_loss)
  -- transfer final state to initial state (BPTT)
  self.network.init_state_global = self.rnn_state[#self.rnn_state]

  if not train_loss then 
    -- renormalize gradients (cliffing gradients)
    local grad_norm, shrink_factor
    if self.opt.hsm == 0 then
      grad_norm = self.network.grad_params:norm()
    else
      grad_norm = torch.sqrt(self.network.grad_params:norm()^2 + self.network.hsm_grad_params:norm()^2)
    end
    if grad_norm > self.opt.max_grad_norm then
      shrink_factor = self.opt.max_grad_norm / grad_norm
      self.network.grad_params:mul(shrink_factor)
      if self.opt.hsm > 0 then self.network.hsm_grad_params:mul(shrink_factor) end
    end
    -- update parameters
    self.network.params:add(self.network.grad_params:mul(-lr))
    if self.opt.hsm > 0 then self.network.hsm_params:add(self.network.hsm_grad_params:mul(-lr)) end
    return nil, nil 
  else -- adam
    local optim_state = {learningRate = 0.0002, beta1 = 0.9, beta2 = 0.999}
    local params, loss = self:adam(train_loss, optim_state)
    return params, loss[1]
  end
end

-- evaluate the loss over an entire split
function Optimizer:evaluate(valid_batcher, max_batches)
  local split_size = valid_batcher.split_nums -- number of the data whcih each batcher has to read by the batch
  if max_batches ~= nil then split_size = math.min(max_batches, split_size) end

  print('evaluating loss over ' .. split_size .. " splits")

  if self.opt.hsm > 0 then self.network.criterion:change_bias() end
  
  local loss = 0
  -- make the previous state at time 1 to be zeros
  local rnn_state = {[0] = self.network.init_state}

  -- iterate over batches
  local i = 1
  for i=1, split_size do
    -- fetch a batch
    local x, y, x_char = valid_batcher:next_batch()  -- from train
    if self.opt.gpuid >= 0 and self.opt.opencl == 0 then -- ship the input arrays to GPU
      -- have to convert to float because integers can't be cuda()'d
      x = x:float():cuda()
      y = y:float():cuda()
      x_char = x_char:float():cuda()
    end
    if self.opt.gpuid >= 0 and self.opt.opencl == 1 then -- ship the input arrays to GPU
      x = x:cl()
      y = y:cl()
      x_char = x_char:cl()
    end

    -- forward pass
    local t = 1
    for t=1, self.opt.seq_length do
      self.network.clones.rnn[t]:evaluate() -- for dropout proper functioning
      local fwd_state = self.network.clones.rnn[t]:forward(self:get_input(x, x_char, t, rnn_state[t-1]))
      rnn_state[t] = {}
      local x = 1
      for x=1, #self.network.init_state do table.insert(rnn_state[t], fwd_state[x]) end
      local prediction = fwd_state[#fwd_state] 
      loss = loss + self.network.clones.criterion[t]:forward(prediction, y[{{}, t}])
    end

    -- carry over lstm state
    rnn_state[0] = rnn_state[#rnn_state]
    -- print(i .. '/' .. split_size .. '...')
  end

  loss = loss / self.opt.seq_length / split_size 

  local perp = torch.exp(loss)    
  return perp
end

-- test the loss over an entire split
function Optimizer:test(test_batcher, max_batches)
  local split_size = test_batcher.split_nums -- number of the data whcih each batcher has to read by the batch
  if max_batches ~= nil then split_size = math.min(max_batches, split_size) end

  print('testing loss over ' .. split_size .. " splits")

  if self.opt.hsm > 0 then self.network.criterion:change_bias() end
  
  local loss = 0
  -- make the previous state at time 1 to be zeros
  local rnn_state = {[0] = self.network.init_state}

  -- fetch a batch
  local x, y, x_char = test_batcher:next_batch()  -- from train
  if self.opt.gpuid >= 0 and self.opt.opencl == 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
    x_char = x_char:float():cuda()
  end
  if self.opt.gpuid >= 0 and self.opt.opencl == 1 then -- ship the input arrays to GPU
    x = x:cl()
    y = y:cl()
    x_char = x_char:cl()
  end

  self.network.rnn:evaluate() -- just need one clone
  -- iterate over batches
  local t = 1
  for t=1, x:size(2) do
    local fwd_state = self.network.rnn:forward(self:get_input(x, x_char, t, rnn_state[0]))
    rnn_state[0] = {}
    local i = 1
    for i=1, #self.network.init_state do table.insert(rnn_state[0], fwd_state[i]) end
    local prediction = fwd_state[#fwd_state] 
    loss = loss + self.network.criterion:forward(prediction, y[{{}, t}])
  end

  loss = loss / x:size(2)

  local perp = torch.exp(loss)    
  return perp
end

-- save checkpoints
function Optimizer:save(savefile, cpu)
  print('saving checkpoint to ' .. savefile)

  local checkpoint = {}
  checkpoint.opt = self.opt
  checkpoint.indexer = self.indexer
  checkpoint.network = self.network
  torch.save(savefile, checkpoint)

  print('saved to ' .. savefile)
end

-- stolen from torch.optim
function Optimizer:adam(loss, config, state)
  local x = self.network.params
  local dfdx = self.network.grad_params
  local fx = loss

  -- (0) get/update state
  local config = config or {}
  local state = state or config
  local lr = config.learningRate or 0.001

  local beta1 = config.beta1 or 0.9
  local beta2 = config.beta2 or 0.999
  local epsilon = config.epsilon or 1e-8

  -- (1) evaluate f(x) and df/dx
  --local fx, dfdx = opfunc(x)

  -- Initialization
  state.t = state.t or 0
  -- Exponential moving average of gradient values
  state.m = state.m or x.new(dfdx:size()):zero()
  -- Exponential moving average of squared gradient values
  state.v = state.v or x.new(dfdx:size()):zero()
  -- A tmp tensor to hold the sqrt(v) + epsilon
  state.denom = state.denom or x.new(dfdx:size()):zero()

  state.t = state.t + 1
    
  -- Decay the first and second moment running average coefficient
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

  state.denom:copy(state.v):sqrt():add(epsilon)

  local biasCorrection1 = 1 - beta1^state.t
  local biasCorrection2 = 1 - beta2^state.t
  local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
  -- (2) update x
  x:addcdiv(-stepSize, state.m, state.denom)

  -- return x*, f(x) before optimization
  return x, {fx}
end

-- start optimization
function Optimizer:train(train_batcher, valid_batcher)
  local train_losses = {}
  local val_losses = {}

  local lr = self.opt.learning_rate -- starting learning rate which will be decayed
  local split_size = train_batcher.split_nums -- number of the data whcih each batcher has to read by the batch
  local iterations = self.opt.max_epochs * split_size

  print('traing loss over ' .. split_size .. " splits")

  -- zero-padding vector is always zero
  if self.network.layer.char_vecs ~= nil then self.network.layer.char_vecs.weight[1]:zero() end
  
  local epoch = 1
  local i = 1
  local progress = 0
  local progress_point = 1

  torch.manualSeed(os.time()) -- it's a trick to show up the progress naturally between 0 to 1

  for epoch=1, self.opt.max_epochs do
    print(string.format("#epoch[%d/%d]", epoch, self.opt.max_epochs))

    -- progress_point will be thrown down between 1 and split_size
    progress_point = (epoch == self.opt.max_epochs) and split_size or torch.random(split_size)

    -- forward and then, backward propagation with parameter update
    for i=1, split_size do

-- ################ feval(params) ######################
      -- initialize gradients before forward propagation every split
      self:initialize_grad_parameters()
      -- get minibatch
      local x, y, x_char = train_batcher:next_batch()  -- from train
      if self.opt.gpuid >= 0 and self.opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        x_char = x_char:float():cuda()
      end
      if self.opt.gpuid >= 0 and self.opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
        x_char = x_char:cl()
      end

      -- forward pass
      local context, predictions, train_loss = self:forward(x, y, x_char)
      train_losses[#train_losses+1] = train_loss
      -- backward pass
      local gradients = self:backward(x, y, x_char, predictions)
      --self:update_parameters(lr, train_loss) -- adam
      self:update_parameters(lr) -- default
-- ################ feval(params) ######################

      -- zero-padding vector is always zero
      if self.network.layer.char_vecs ~= nil then 
        self.network.layer.char_vecs.weight[1]:zero() 
        self.network.layer.char_vecs.gradWeight[1]:zero() 
      end

      --print(string.format("--split[%d/%d] in epoch[%d/%d]", i, split_size, epoch, self.opt.max_epochs))
      if i == progress_point then progress = (((epoch-1) * split_size) + i) / (self.opt.max_epochs * split_size) end
    end

    -- evaluate loss on validation data
    local val_loss = self:evaluate(valid_batcher) -- validation
    val_losses[#val_losses+1] = val_loss

    -- decay learning rate every epoch after evaluating the perflexities from validation data
    if #val_losses > 2 and val_losses[#val_losses-1] - val_losses[#val_losses] < self.opt.decay_when then
      lr = lr * self.opt.learning_rate_decay
    end

    -- print progress
    if epoch % self.opt.print_every == 0 then
      print(string.format("--progress = %.4f, train loss = %6.4f", progress, train_losses[#train_losses]))
    end

    -- save checkpoint
    if epoch == self.opt.max_epochs or epoch % self.opt.save_every == 0 then
      local savefile = string.format('%s/%s_model_epoch_%.2f_%.2f.t7', self.opt.checkpoint_dir, self.opt.savefile, epoch, val_loss)
      self:save(savefile)
    end

    -- garbage collection every epoch
    collectgarbage()
  end

end
