function test_channel_pooling_layer(backend::Backend, pooling::PoolingFunction, tensor_dim::Int, n_input, T, eps)
  println("-- Testing ChannelPooling($(typeof(pooling))) on $(typeof(backend)){$T}...")

  dims = [abs(rand(Int, tensor_dim)) % 7 + 7 for i = 1:n_input]
  op_dim = max(abs(rand(Int)) % tensor_dim, 1)
  pad = (2,2)
  kernel = 3
  stride = 2

  println("    > Setup (pool along dimension $op_dim for $tensor_dim-D tensors)")

  layer = ChannelPoolingLayer(kernel=kernel, stride=stride, pad=pad, pooling=pooling,
      tops=Array(Symbol,n_input), bottoms=Array(Symbol,n_input), channel_dim=op_dim)

  input = [rand(T, dim...) for dim in dims]
  inputs = Blob[make_blob(backend, x) for x in input]
  diffs = Blob[make_blob(backend, x) for x in input]

  state = setup(backend, layer, inputs, diffs)

  println("    > Forward")
  forward(backend, state, inputs)

  payloads = Array(Any, n_input)
  for i = 1:n_input
    expected_output, payloads[i] = channel_pooling_forward(state, i, input[i], op_dim)
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
    expected_output = channel_pooling_backward(state, i, input[i], top_diff[i], payloads[i], op_dim)
    got_output = to_array(diffs[i])
    @test all(-eps .< expected_output - got_output .< eps)
  end

  shutdown(backend, state)
end

function channel_pooling_forward(state, i, input::Array, op_dim)
  dim_pre, dim_pool, dim_post = split_dims(input, op_dim)
  dim_pooled = size(state.blobs[i], op_dim)

  output = zeros(eltype(input), size(state.blobs[i]))
  if isa(state.layer.pooling, Pooling.Max)
    mask = similar(output, Int)
  end

  canonical_input = reshape(input, dim_pre, dim_pool, dim_post)
  canonical_output = reshape(output, dim_pre, dim_pooled, dim_post)
  if isa(state.layer.pooling, Pooling.Max)
    canonical_mask = reshape(mask, dim_pre, dim_pooled, dim_post)
  end

  for n = 1:dim_post
    for pc = 1:dim_pooled
      cstart = (pc-1)*state.layer.stride - state.layer.pad[1] + 1
      cend = min(cstart + state.layer.kernel - 1, size(input, op_dim))
      cstart = max(1, cstart)

      region = canonical_input[:,cstart:cend,n]
      if isa(state.layer.pooling, Pooling.Max)
        maxval, maxidx = findmax(region, 2)
        canonical_output[:,pc,n] = maxval
        canonical_mask[:,pc,n] = maxidx
      elseif isa(state.layer.pooling, Pooling.Mean)
        canonical_output[:,pc,n] = sum(region, 2) / state.layer.kernel
      else
        error("Unknown pooling $(state.layer.pooling)")
      end
    end
  end

  if isa(state.layer.pooling, Pooling.Max)
    return (output, mask)
  else
    return (output, Void())
  end
end

function channel_pooling_backward(state, i, input::Array, diff::Array, payload::Any, op_dim)
  dim_pre, dim_pool, dim_post = split_dims(input, op_dim)
  dim_pooled = size(state.blobs[i], op_dim)

  gradient = zeros(eltype(input), size(input))
  canonical_input = reshape(input, dim_pre, dim_pool, dim_post)
  canonical_gradient = reshape(gradient, dim_pre, dim_pool, dim_post)
  canonical_diff = reshape(diff, dim_pre, dim_pooled, dim_post)
  if isa(state.layer.pooling, Pooling.Max)
    canonical_mask = reshape(payload, dim_pre, dim_pooled, dim_post)
  end
  for n = 1:dim_post
    for pc = 1:dim_pooled
      cstart = (pc-1)*state.layer.stride - state.layer.pad[1] + 1
      cend = min(cstart + state.layer.kernel - 1, size(input, op_dim))
      cstart = max(1, cstart)

      if isa(state.layer.pooling, Pooling.Max)
        region = sub(canonical_gradient,1:dim_pre,cstart:cend,n)
        maxidx = canonical_mask[:,pc,n]
        region[vec(maxidx)] += vec(canonical_diff[:,pc,n])
      elseif isa(state.layer.pooling, Pooling.Mean)
        for c = cstart:cend
          canonical_gradient[:,c,n] += canonical_diff[:,pc,n] / state.layer.kernel
        end
      else
        error("Unknown pooling $(state.layer.pooling)")
      end
    end
  end
  return gradient
end

function test_channel_pooling_layer(backend::Backend, pooling::PoolingFunction, n_input, T, eps)
  for i = 2:6
    test_channel_pooling_layer(backend, pooling, i, n_input, T, eps)
  end
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
