--[[
Convert GPU model to CPU model

Woohyun Kim (deepcoord@gmail.com)
--]]
-- torch7
require('torch')
require('nn')
require('nngraph')
require('optim')
require('lfs')

-- util
require('util.parser')
require('util.filereader')
require('util.wordindexer')
require('util.inputloader')
require('util.batcher')
require('util.tensorbatcher')

require('util.Squeeze')

-- model
require('model.RNN')
require('model.LSTM')
require('model.GRU')
require('model.HighwayMLP')
require('model.TDNN')
require('model.LSTMTDNN')
-- criterion
require('model.HSMClass')
require('model.HLogSoftMax')

-- network & optimizer
require('network')
require('optimizer')

--BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert GPU model to CPU model')
cmd:text()
cmd:text('Options')
-- model
cmd:option('-checkpoint','cv/checkpoint.t7','GPU model to convert')
-- GPU
cmd:option('-gpuid',-1,'GPU device')

opt = cmd:parse(arg)

-- GPU
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpuid + 1)
end

print('loading ' .. opt.checkpoint .. ' for converting')
local checkpoint = torch.load(opt.checkpoint)

local checkpoint_4cpu = {}
checkpoint_4cpu.opt = checkpoint.opt
checkpoint_4cpu.opt.gpuid = -1
checkpoint_4cpu.indexer = checkpoint.indexer
local network = Network(checkpoint_4cpu.opt) -- create a new network
network.opt = checkpoint_4cpu.opt
network.rnn = checkpoint.network.rnn:double()
network.criterion = checkpoint.network.criterion:double()
local init_state = {}
for k,v in pairs(checkpoint.network.init_state) do init_state[k] = v:double():clone() end
local init_state_global = {}
for k,v in pairs(checkpoint.network.init_state_global) do init_state_global[k] = v:double():clone() end
network.init_state = init_state
checkpoint_4cpu.network = network

local savefile = string.gsub(opt.checkpoint, paths.extname(opt.checkpoint), "4cpu.t7")
print('saving checkpoint to ' .. savefile)
torch.save(savefile, checkpoint_4cpu)
print('saved.')
