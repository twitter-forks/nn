function nn.LayerNormalization(nOutput, bias, eps, affine)
  eps = eps or 1e-10
  affine = affine or true
  bias = bias or 0

  local layer = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Sequential()
        :add(nn.Mean(1, 1))
        :add(nn.Replicate(nOutput,1,1))))
    :add(nn.CSubTable())
    :add(nn.Normalize(2, eps))
    :add(nn.MulConstant(torch.sqrt(nOutput)))

  if affine then
    local biasTransform = nn.Add(nOutput, false)
    biasTransform.bias:fill(bias)
    local gainTransform = nn.CMul(nOutput)
    gainTransform.weight:fill(1.)
    layer:add(gainTransform)
    layer:add(biasTransform)
  end

  return layer
end
