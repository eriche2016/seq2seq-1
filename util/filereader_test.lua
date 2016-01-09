--[[
File Reader Test

Woohyun Kim(deepcoord@gmail.com)
--]]

require("parser")
require("filereader")

cmd = torch.CmdLine()
cmd:option('-file', 'file', 'file to read')

opt = cmd:parse(arg)

if opt.file == "" then
  print("Usage: th filereader_test.lua -file <file>")
  os.exit()
end

if not paths.filep(opt.file) then
  print("Cannot find " .. opt.file)
  print("Usage: th filereader_test.lua -file <file>")
  os.exit()
end

tokens = {}
tokens.SEP = "%s" -- word separator
tokens.EOP = "<eop>" -- end of a pair
tokens.EOS = "<eos>" -- end of a sentence

local reader = FileReader(opt.file, tokens)

print("================== line ====================")
local count = 1
while 1 do
  local line = reader:readline()
  if line == nil then break end
  print(string.format("line[%d]\t\t%s", count, line))
  count = count + 1
end
reader:close() -- close here to clerify (even though it has internally closed)

print("================== word ====================")
count = 1
while 1 do
  local words = reader:readwords()
  if words == nil then break end

  for i, word in ipairs(words) do
    print(string.format("word[%d][%d]\t\t%s", count, i, word))
  end

  count = count + 1
end
reader:close() -- close here to clerify (even though it has internally closed)
