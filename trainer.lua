--[[
Trainer for Sequence to Seqeunce Learning

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

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a sequence to sequence model')
cmd:text()
cmd:text('Options')

-- data
cmd:option('-data_dir','data/dialog', 'data directory containing train.txt/valid.txt/test.txt')

-- model
cmd:option('-model', 'lstmtdnn', 'lstmtdnn, highwaymlp, tdnn, lstm, gru, or rnn, but for now only lstm is supported')
cmd:option('-use_words', 1, 'use words (1=yes)')
cmd:option('-use_chars', 0, 'use characters (1=yes)')
cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-highway_layers', 2, 'number of highway layers in the LSTM-TDNN')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-char_vec_size', 15, 'dimensionality of character embeddings')
cmd:option('-feature_maps', '{50,100,150,200,200,200,200}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{1,2,3,4,5,6,7}', 'convolution network kernel widths')
cmd:option('-dropout',0.5,'dropout for regularization, 0 = no dropout')
cmd:option('-min_freq', 1, 'minimum frequences for building vocabulary')
cmd:option('-EOS', '<eos>', 'end of sequences, <eos> symbol("\n")')
cmd:option('-EOP', '<eop>', 'end of pairs in ssequence, <eop> symbol("\t")')

-- optimization
cmd:option('-seq_length',35,'number of timesteps to unroll for')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-hsm',0,'number of clusters to use for hsm. 0 = normal softmax, -1 = use sqrt(|V|)')
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-batch_norm', 0, 'use batch normalization over input embeddings (1=yes)')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-max_epochs',100,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')

-- bookkeeping
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-print_every',1,'print out the loss every epoch')
cmd:option('-save_every',1,'save checkpoint every n epochs')
cmd:option('-savefile','seq2seq','filename to autosave the checkpont to. Will be inside checkpoint_dir/')

-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use, -1 = use CPU')
cmd:option('-cudnn',0,'use cudnn (1=yes) to greatly speed up convolutions')
cmd:option('-opencl',0,'use CUDA (instead of OpenCL)')

cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- global constants for certain tokens
opt.tokens = {}
opt.tokens.EOP = opt.EOP -- end of pairs in sequences, <eop>
opt.tokens.EOS = opt.EOS -- end of sequences, <eos>
opt.tokens.UNK = "<unk>" -- unk word token
opt.tokens.SEP = "%s" -- word separator pattern
opt.tokens.START = '{' -- start-of-word token
opt.tokens.END = '}' -- end-of-word token
opt.tokens.ZEROPAD = ' ' -- zero-pad token 

torch.manualSeed(os.time())

-- initialize CUDA on GPU (cunn/cutorch)
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    --cutorch.manualSeed(cutorch.initialSeed())
    cutorch.manualSeed(os.time())
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end
-- initialize OpenCL on GPU (clnn/cltorch) 
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    print('using OpenCL on GPU ' .. opt.gpuid .. '...')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    --cltorch.manualSeed(cltorch.initialSeed())
    torch.manualSeed(os.time())
  else
    print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
    print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- start re-training from a specific checkpoint
local checkpoint = {}
local retrain = false
if path.exists(opt.checkpoint) then
   print('loading ' .. opt.checkpoint .. ' for re-training')
   checkpoint = torch.load(opt.checkpoint)

   for k, v in pairs(opt) do
     if type(opt[k]) ~= 'table' then
       if opt[k] ~= checkpoint.opt[k] then
         print(string.format("change %s to %s ?", checkpoint.opt[k], opt[k]))
         local yes_or_no = io.read()
         yes_or_no = string.lower(yes_or_no)
         if yes_or_no == 'yes' or yes_or_no == 'y' then
           checkpoint.opt[k] = opt[k]
         end
       end
     else
       print("table comaprison will be developed later")
       checkpoint.opt[k] = opt[k]
     end
   end

   opt = checkpoint.opt
   retrain = true
end

-- some housekeeping
loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes
assert(#opt.kernels == #opt.feature_maps, '#kernels has to equal #feature maps')

-- prepare input data for training
local path = require('pl.path')
if not path.exists(opt.data_dir) then
  print("Usage: th main.lua -data_dir <data_dir>")
  os.exit()
end

-- make sure input files exist
local train_file = path.join(opt.data_dir, 'train.txt')
local valid_file = path.join(opt.data_dir, 'valid.txt')
local test_file = path.join(opt.data_dir, 'test.txt')
if not paths.filep(train_file) then print("Cannot find " .. train_file); os.exit() end
if not paths.filep(valid_file) then print("Cannot find " .. valid_file); os.exit() end
if not paths.filep(test_file) then print("Cannot find " .. test_file); os.exit() end

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- build up vocabularies from train/valid/test
local indexer
if retrain then 
  indexer = checkpoint.indexer 
else
  indexer = WordIndexer(opt.tokens, opt.min_freq)
  indexer:add(train_file)
  indexer:add(valid_file)
  indexer:add(test_file)
end 
indexer:stats()

-- ready for making tensor batches 
local train_batcher = TensorBatcher(indexer, opt.batch_size, opt.seq_length)
train_batcher:open(train_file)
train_batcher:make_batches()
local valid_batcher = TensorBatcher(indexer, opt.batch_size, opt.seq_length)
valid_batcher:open(valid_file)
valid_batcher:make_batches()
local test_batcher = TensorBatcher(indexer, opt.batch_size, opt.seq_length)
test_batcher:open(test_file)
test_batcher:make_batches()
print(string.format("train batches = %d, valid batches = %d, test batches = %d", train_batcher.split_nums, valid_batcher.split_nums, test_batcher.split_nums))


-- create network layers and architectures 
local network 
if retrain then
  network = checkpoint.network
else
  network = Network(opt)
  network:make_model(indexer)
  network:clone_model()
end

-- optimize the network
local optimizer = Optimizer(indexer, network)
optimizer:train(train_batcher, valid_batcher)

local test_perp = optimizer:test(test_batcher)
print('Perplexity on testset: ' .. test_perp)


