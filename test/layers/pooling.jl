function test_pooling_layer(backend::Backend, pooling::PoolingFunction, has_padding::Bool, n_input, T, eps)
  println("-- Testing Pooling($(typeof(pooling))) $(has_padding? "with padding":"") on $(typeof(backend)){$T}...")
  println("    > Setup")

  if has_padding
    input_w   = 16
    input_h   = 10
    padding   = (2,2)
  else
    input_w   = 18
    input_h   = 12
    padding   = (0, 0)
  end
  input_chann = 3
  input_num   = 24
  kernel_w    = 3
  kernel_h    = 4
  stride_w    = 1
  stride_h    = 2

  dims = [rand(12:16, 4) for i =1:n_input]
  dims[1] = [input_w, input_h, input_chann, input_num]
  if isa(backend, AbstractGPUBackend)
    for i = 1:n_input
      # cuDNN pooling have different behavior on the boundary.
      # For mean pooling, cuDNN restricts the pooling size when it runs beyond the boundary.
      # Thus at the boundary, the mean values are larger than Mocha's default behavior,
      # because in Mocha's CPU implementation, the mean will count zeros beyond the the
      # boundary. I think this minor difference at the boundary is not super important.
      # So instead of messing around the implementation of GPU pooling (e.g. add explicit
      # padding when the pooling could run beyond the boundary), I use a workaround
      # in the unit test to make sure that the pooling always ends exactly at the boundary
      # in our unit tests.
      dims[i][1] -= (dims[i][1] + 2*padding[1] - kernel_w) % stride_w
      dims[i][2] -= (dims[i][2] + 2*padding[2] - kernel_h) % stride_h
    end
  end

  layer = PoolingLayer(kernel=(kernel_w,kernel_h), stride=(stride_w,stride_h), pad=padding,
      tops=Array{Symbol}(n_input), bottoms=Array{Symbol}(n_input), pooling=pooling)

  input = [rand(T, dims[i]...) for i = 1:n_input]
  inputs = Blob[make_blob(backend, x) for x in input]
  diffs = Blob[make_blob(backend, x) for x in input]

  state = setup(backend, layer, inputs, diffs)

  println("    > Forward")
  forward(backend, state, inputs)

  payloads = Array{Any}(n_input)
  for i = 1:n_input
    expected_output, payloads[i] = pooling_forward(state, i, input[i])
    got_output = similar(expected_output)
    copy!(got_output, state.blobs[i])
    @test all(-eps .< expected_output - got_output .< eps)
  end

  println("    > Backward")
  top_diff = [rand(T, size(state.blobs[i])) for i = 1:n_input]
  for i = 1:n_input
    copy!(state.blobs_diff[i], top_diff[i])
  end

  backward(backend, state, inputs, diffs)

  for i = 1:n_input
    expected_grad = pooling_backward(state, i, input[i], top_diff[i], payloads[i])
    got_grad = similar(expected_grad)
    copy!(got_grad, diffs[i])
    @test all(-eps .< expected_grad - got_grad .< eps)
  end

  shutdown(backend, state)
end

function pooling_forward(state, i, input::Array)
  width, height, channels, num = size(input)
  pooled_width = get_width(state.blobs[i])
  pooled_height = get_height(state.blobs[i])
  kernel_size = state.layer.kernel[1] * state.layer.kernel[2]

  output = zeros(eltype(input), pooled_width, pooled_height, channels, num)
  if isa(state.layer.pooling, Pooling.Max)
    mask = similar(output, Int)
  end

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph-1)*state.layer.stride[2] - state.layer.pad[2] + 1
          wstart = (pw-1)*state.layer.stride[1] - state.layer.pad[1] + 1
          hend = min(hstart + state.layer.kernel[2] - 1, height)
          wend = min(wstart + state.layer.kernel[1] - 1, width)
          hstart = max(1, hstart)
          wstart = max(1, wstart)

          region = view(input, wstart:wend, hstart:hend, c, n)
          if isa(state.layer.pooling, Pooling.Max)
            index = indmax(region)
            mask[pw, ph, c, n] = index # note this is local index in region
            output[pw, ph, c, n] = region[index]
          elseif isa(state.layer.pooling, Pooling.Mean)
            output[pw, ph, c, n] = sum(region) / kernel_size
          else
            error("Unknown pooling $(state.layer.pooling)")
          end
        end
      end
    end
  end

  if isa(state.layer.pooling, Pooling.Max)
    return (output, mask)
  else
    return (output, nothing)
  end
end

function pooling_backward(state, i, input::Array, diff::Array, payload::Any)
  width, height, channels, num = size(input)
  pooled_width = get_width(state.blobs[i])
  pooled_height = get_height(state.blobs[i])
  kernel_size = state.layer.kernel[1] * state.layer.kernel[2]

  gradient = zeros(eltype(input), size(input))

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph-1)*state.layer.stride[2] - state.layer.pad[2] + 1
          wstart = (pw-1)*state.layer.stride[1] - state.layer.pad[1] + 1
          hend = min(hstart + state.layer.kernel[2] - 1, height)
          wend = min(wstart + state.layer.kernel[1] - 1, width)
          hstart = max(1, hstart)
          wstart = max(1, wstart)

          region = view(gradient, wstart:wend, hstart:hend, c, n)
          if isa(state.layer.pooling, Pooling.Max)
            index = payload[pw, ph, c, n]
            region[index] += diff[pw, ph, c, n]
          elseif isa(state.layer.pooling, Pooling.Mean)
            region[:] += diff[pw, ph, c, n] / kernel_size
          else
            error("Unknown pooling $(state.layer.pooling)")
          end
        end
      end
    end
  end

  return gradient
end

function test_pooling_layer(backend::Backend, n_input, T, eps)
  test_pooling_layer(backend, Pooling.Max(), false, n_input, T, eps)
  test_pooling_layer(backend, Pooling.Mean(), false, n_input, T, eps)
  test_pooling_layer(backend, Pooling.Max(), true, n_input, T, eps)
  test_pooling_layer(backend, Pooling.Mean(), true, n_input, T, eps)
end

function test_pooling_layer(backend::Backend)
  test_pooling_layer(backend, 4, Float64, 1e-7)
  test_pooling_layer(backend, 2, Float32, 1e-3)
end

if test_cpu
  test_pooling_layer(backend_cpu)
end
if test_gpu
  test_pooling_layer(backend_gpu)
end
