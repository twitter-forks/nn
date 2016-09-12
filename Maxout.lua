function nn.Maxout(inputSize, outputSize, maxoutNumber, preprocess)
  local ret = nn.Sequential()
  ret:add(nn.Linear(inputSize, outputSize * maxoutNumber))
  ret:add(nn.View(maxoutNumber, outputSize):setNumInputDims(1))
  if preprocess then
    ret:add(preprocess)
  end
  ret:add(nn.Max(1, 2))
  return ret
end
