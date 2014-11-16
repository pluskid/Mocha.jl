function test_channel_pooling_layer(sys::System, pooling::PoolingFunction)
  println("-- Testing Pooling($(typeof(pooling))) on $(typeof(sys.backend))...")
  println("    > Setup")

  width, height, channels, num = (2, 3, 7, 1)
  pad = (1,2)
  kernel = 3
  stride = 2
  eps = 1e-7

  layer = ChannelPoolingLayer(kernel=kernel, stride=stride, pad=pad,
      tops=[:top], bottoms=[:bottom], pooling=pooling)

  input_dim = (width, height, channels, num)
  input = rand(input_dim)
  inputs = Blob[make_blob(sys.backend, input)]
  diffs = Blob[make_blob(sys.backend, input)]

  state = setup(sys, layer, inputs, diffs)

  println("    > Forward")
  forward(sys, state, inputs)

  expected_output, payload = channel_pooling_forward(state, input)
  got_output = similar(expected_output)
  copy!(got_output, state.blobs[1])
  @test all(-eps .< expected_output-got_output .< eps)
end

function channel_pooling_forward(state, input::Array)
  width, height, channels, num = size(input)
  pooled_chann = get_chann(state.blobs[1])

  output = zeros(width, height, pooled_chann, num)
  if isa(state.layer.pooling, Pooling.Max)
    mask = similar(output, Int)
  end

  for n = 1:num
    for pc = 1:pooled_chann
      cstart = max(1, (pc-1)*state.layer.stride - state.layer.pad[1] + 1)
      cend = min(cstart + state.layer.kernel - 1, channels)
      region = input[:,:,cstart:cend, n]
      if isa(state.layer.pooling, Pooling.Max)
        maxval, maxidx = findmax(region, 3)
        output[:,:,pc,n] = maxval
        mask[:,:,pc,n] = maxidx
      elseif isa(state.layer.pooling, Pooling.Mean)
        output[:,:,pc,n] = sum(region, 3) / state.layer.kernel
      else
        error("Unknown pooling $(state.layer.pooling)")
      end
    end
  end

  if isa(state.layer.pooling, Pooling.Max)
    return (output, mask)
  else
    return (output, nothing)
  end
end

function test_channel_pooling_layer(sys::System)
  test_channel_pooling_layer(sys, Pooling.Max())
  test_channel_pooling_layer(sys, Pooling.Mean())
end

if test_cpu
  test_channel_pooling_layer(sys_cpu)
end
if test_cudnn
  test_channel_pooling_layer(sys_cudnn)
end
