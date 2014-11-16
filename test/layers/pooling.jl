function test_pooling_layer(sys::System, pooling::PoolingFunction, has_padding::Bool)
  println("-- Testing Pooling($(typeof(pooling))) on $(typeof(sys.backend))...")
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
  eps         = 1e-10

  layer = PoolingLayer(kernel=(kernel_w,kernel_h), stride=(stride_w,stride_h), pad=padding,
      tops=[:output], bottoms=[:input], pooling=pooling)

  input_dims = (input_w, input_h, input_chann, input_num)
  input = rand(input_dims)
  inputs = Blob[make_blob(sys.backend, Float64, input_dims)]
  diffs = Blob[make_blob(sys.backend, Float64, input_dims)]
  copy!(inputs[1], input)

  state = setup(sys, layer, inputs, diffs)

  println("    > Forward")
  forward(sys, state, inputs)

  expected_output, payload = pooling_forward(state, input)
  got_output = similar(expected_output)
  copy!(got_output, state.blobs[1])
  @test all(-eps .< expected_output - got_output .< eps)

  println("    > Backward")
  top_diff = rand(size(state.blobs[1]))
  copy!(state.blobs_diff[1], top_diff)

  backward(sys, state, inputs, diffs)

  expected_grad = pooling_backward(state, input, top_diff, payload)
  got_grad = similar(expected_grad)
  copy!(got_grad, diffs[1])
  @test all(-eps .< expected_grad - got_grad .< eps)
end

function pooling_forward(state, input::Array)
  width, height, channels, num = size(input)
  pooled_width = get_width(state.blobs[1])
  pooled_height = get_height(state.blobs[1])
  kernel_size = state.layer.kernel[1] * state.layer.kernel[2]

  output = zeros(pooled_width, pooled_height, channels, num)
  if isa(state.layer.pooling, Pooling.Max)
    mask = similar(output, Int)
  end

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = max(1, (ph-1)*state.layer.stride[2] - state.layer.pad[2] + 1)
          wstart = max(1, (pw-1)*state.layer.stride[1] - state.layer.pad[1] + 1)
          hend = min(hstart + state.layer.kernel[2] - 1, height)
          wend = min(wstart + state.layer.kernel[1] - 1, width)

          region = sub(input, wstart:wend, hstart:hend, c, n)
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

function pooling_backward(state, input::Array, diff::Array, payload::Any)
  width, height, channels, num = size(input)
  pooled_width = get_width(state.blobs[1])
  pooled_height = get_height(state.blobs[1])
  kernel_size = state.layer.kernel[1] * state.layer.kernel[2]

  gradient = zeros(size(input))

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = max(1, (ph-1)*state.layer.stride[2] - state.layer.pad[2] + 1)
          wstart = max(1, (pw-1)*state.layer.stride[1] - state.layer.pad[1] + 1)
          hend = min(hstart + state.layer.kernel[2] - 1, height)
          wend = min(wstart + state.layer.kernel[1] - 1, width)

          region = sub(gradient, wstart:wend, hstart:hend, c, n)
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

function test_pooling_layer(sys::System)
  test_pooling_layer(sys, Pooling.Max(), false)
  test_pooling_layer(sys, Pooling.Mean(), false)
  if !isa(sys.backend, AbstractCuDNNBackend)
    test_pooling_layer(sys, Pooling.Max(), true)
    test_pooling_layer(sys, Pooling.Mean(), true)
  end
end

if test_cpu
  test_pooling_layer(sys_cpu)
end
if test_cudnn
  test_pooling_layer(sys_cudnn)
end
