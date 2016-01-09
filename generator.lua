--[[
Generate some possible sequences based on the given sequences

Woohyun Kim (deepcoord@gmail.com)
]]--

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

-- encoder/decoder
require('encoder')
require('decoder')


cmd = torch.CmdLine()
cmd:text('Options')

-- data
cmd:option('-model', 'cv/model.t7', 'checkpoint file')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use, -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-cudnn', 0,'use cudnn (1 = yes, 0 = no)')
-- reversed mode
cmd:option('-reverse', 0,'use input in reverse mode')
-- options
cmd:option('-length', '25', 'the length of text to generate')
cmd:option('-temperature', 1, 'temperature of sampling (1 = dynamic, 0 = static)')
-- web server
cmd:option('-server', 0, 'run as a web server (0> = server on port)')

cmd:text()

-- parse input params
opt2 = cmd:parse(arg)

-- initialize CUDA on GPU (cunn/cutorch)
if opt2.gpuid >= 0 and opt2.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    cutorch.setDevice(opt2.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    --cutorch.manualSeed(cutorch.initialSeed())
    cutorch.manualSeed(os.time())
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt2.gpuid = -1 -- overwrite user setting
  end
end
-- initialize OpenCL on GPU (clnn/cltorch) 
if opt2.gpuid >= 0 and opt2.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    print('using OpenCL on GPU ' .. opt2.gpuid .. '...')
    cltorch.setDevice(opt2.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    --cltorch.manualSeed(cltorch.initialSeed())
    torch.manualSeed(os.time())
  else
    print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
    print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
    print('Falling back on CPU mode')
    opt2.gpuid = -1 -- overwrite user setting
  end
end

HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

torch.manualSeed(os.time())

-- load the pretrained model to generate sequences
local checkpoint = {}
if path.exists(opt2.model) then
  print('loading the model ' .. opt2.model)
  checkpoint = torch.load(opt2.model)
else 
  print('cannot load the model ' .. opt2.model)
  os.exit()
end

opt = checkpoint.opt
--print('opt: ' )
--print(opt)

if opt2.gpuid ~= opt.gpuid then
  print("Model environment(GPU/CPU) was different")
  opt.gpuid = opt2.gpuid
end

local indexer = checkpoint.indexer
local network = checkpoint.network
-- turn off dropout and keeping timesteps
if opt2.temperature == 0 then network.rnn:evaluate() end
local encoder = Encoder(indexer, network)
local decoder = Decoder(indexer, network)

-- run on the command line
if opt2.server == 0 then
  while 1 do
    io.write("You: ")
    local line = io.read()
    if line == nil or line == "" then break end

    -- get a fixed-length vector representation of the given text
    -- and return predicted output by softmax
    local encode,  prediction = encoder:encode(line)
    -- start sampling/argmaxing
    local decode, sequence = decoder:decode(encode, prediction)

    io.write("Musio: ")
    for _, w in pairs(sequence) do
      io.write(w .. " ")
    end
    io.write('\n') io.flush()
  end
-- run as a web service
else
  local json = require('json')
  local app = require('waffle')

  -- process HTTP post method
  app.post('/', function(req, res)
    print("request : " .. req.body)
    local json_req = json.decode(req.body)

    local pattern = json_req.pattern
    if pattern == nil or pattern == "" then pattern = "Hello!" end

    -- acquiring context from the given sequence
    local encode, prediction = encoder:encode(pattern)
    -- sampling/argmaxing the next sequence
    local decode, sequence = decoder:decode(encode, prediction, opt2.length)

    local t = {}
    for _, w in pairs(sequence) do
      table.insert(t, w)
    end
    local template = table.concat(t, " ")

    local response = {}
    response["pattern"] = pattern
    response["template"] = template
    print("response : " .. json.encode(response));

    res.send(json.encode(response))
  end)

  -- run a REST server
  if opt2.server > 0 then
    server_options = {}
    server_options.port = (opt2.server > 1024) and opt2.server or 8080
    app.listen(server_options)
  else
    app.listen()
  end
end
