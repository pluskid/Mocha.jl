function test_channel_pooling_layer(sys::System, pooling::PoolingFunction)
  println("-- Testing Pooling($(typeof(pooling))) on $(typeof(sys.backend))...")
  println("    > Setup")

  width, height, channels, num = (2, 3, 7, 1)
  pad = (2,2)
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

  println("    > Backward")
  top_diff = rand(size(state.blobs[1]))
  copy!(state.blobs_diff[1], top_diff)
  backward(sys, state, inputs, diffs)

  expected_output = channel_pooling_backward(state, input, top_diff, payload)
  got_output = similar(expected_output)
  copy!(got_output, diffs[1])
  @test all(-eps .< expected_output - got_output .< eps)
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

function channel_pooling_backward(state, input::Array, diff::Array, payload::Any)
  width, height, channels, num = size(input)
  pooled_chann = get_chann(state.blobs[1])

  gradient = zeros(width, height, channels, num)
  for n = 1:num
    for pc = 1:pooled_chann
      cstart = max(1, (pc-1)*state.layer.stride - state.layer.pad[1] + 1)
      cend = min(cstart + state.layer.kernel - 1, channels)
      if isa(state.layer.pooling, Pooling.Max)
        region = sub(gradient,1:width,1:height,cstart:cend,n)
        maxidx = payload[:,:,pc,n]
        region[maxidx] += diff[:,:,pc,n]
      elseif isa(state.layer.pooling, Pooling.Mean)
        for c = cstart:cend
          gradient[:,:,c,n] += diff[:,:,pc,n] / state.layer.kernel
        end
      else
        error("Unknown pooling $(state.layer.pooling)")
      end
    end
  end
  return gradient
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
