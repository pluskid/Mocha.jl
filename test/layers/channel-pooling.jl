function test_channel_pooling_layer(backend::Backend, pooling::PoolingFunction, n_input, T, eps)
  println("-- Testing ChannelPooling($(typeof(pooling))) on $(typeof(backend)){$T}...")
  println("    > Setup")

  dims = [abs(rand(Int, 4)) % 7 + 7 for i = 1:n_input]
  pad = (2,2)
  kernel = 3
  stride = 2

  layer = ChannelPoolingLayer(kernel=kernel, stride=stride, pad=pad, pooling=pooling,
      tops=Array(Symbol,n_input), bottoms=Array(Symbol,n_input))

  input = [rand(T, dim...) for dim in dims]
  inputs = Blob[make_blob(backend, x) for x in input]
  diffs = Blob[make_blob(backend, x) for x in input]

  state = setup(backend, layer, inputs, diffs)

  println("    > Forward")
  forward(backend, state, inputs)

  payloads = Array(Any, n_input)
  for i = 1:n_input
    expected_output, payloads[i] = channel_pooling_forward(state, i, input[i])
    got_output = to_array(state.blobs[i])
    @test all(-eps .< expected_output-got_output .< eps)
  end

  println("    > Backward")
  top_diff = [rand(T, size(state.blobs[i])) for i = 1:n_input]
  for i = 1:n_input
    copy!(state.blobs_diff[i], top_diff[i])
  end
  backward(backend, state, inputs, diffs)

  for i = 1:n_input
    expected_output = channel_pooling_backward(state, i, input[i], top_diff[i], payloads[i])
    got_output = to_array(diffs[i])
    @test all(-eps .< expected_output - got_output .< eps)
  end

  shutdown(backend, state)
end

function channel_pooling_forward(state, i, input::Array)
  width, height, channels, num = size(input)
  pooled_chann = get_chann(state.blobs[i])

  output = zeros(eltype(input), width, height, pooled_chann, num)
  if isa(state.layer.pooling, Pooling.Max)
    mask = similar(output, Int)
  end

  for n = 1:num
    for pc = 1:pooled_chann
      cstart = (pc-1)*state.layer.stride - state.layer.pad[1] + 1
      cend = min(cstart + state.layer.kernel - 1, channels)
      cstart = max(1, cstart)

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

function channel_pooling_backward(state, i, input::Array, diff::Array, payload::Any)
  width, height, channels, num = size(input)
  pooled_chann = get_chann(state.blobs[i])

  gradient = zeros(eltype(input), width, height, channels, num)
  for n = 1:num
    for pc = 1:pooled_chann
      cstart = (pc-1)*state.layer.stride - state.layer.pad[1] + 1
      cend = min(cstart + state.layer.kernel - 1, channels)
      cstart = max(1, cstart)

      if isa(state.layer.pooling, Pooling.Max)
        region = sub(gradient,1:width,1:height,cstart:cend,n)
        maxidx = payload[:,:,pc,n]
        region[vec(maxidx)] += vec(diff[:,:,pc,n])
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

function test_channel_pooling_layer(backend::Backend, n_input, T, eps)
  test_channel_pooling_layer(backend, Pooling.Max(), n_input, T, eps)
  test_channel_pooling_layer(backend, Pooling.Mean(), n_input, T, eps)
end

function test_channel_pooling_layer(backend::Backend)
  test_channel_pooling_layer(backend, 1, Float32, 1e-4)
  test_channel_pooling_layer(backend, 3, Float64, 1e-8)
end

if test_cpu
  test_channel_pooling_layer(backend_cpu)
end
if test_gpu
  test_channel_pooling_layer(backend_gpu)
end
