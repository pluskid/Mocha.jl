function test_lrn_layer(backend::Backend, mode::LRNModeType, tensor_dim, T, eps)
  println("-- Testing LRN($(typeof(mode))) on $(typeof(backend)){$T}...")


  dims = tuple(rand(6:11, tensor_dim)...)
  op_dim = max(abs(rand(Int)) % tensor_dim, 1)

  println("    > Setup with dims $dims")

  input = rand(T, dims)
  input_blobs = Blob[make_blob(backend, input)]
  diff_blobs = Blob[make_blob(backend, input)]

  layer = LRNLayer(tops=[:output],bottoms=[:input], mode=mode, channel_dim=op_dim)
  state = setup(backend, layer, input_blobs, diff_blobs)

  @test size(state.blobs[1]) == size(input)

  println("    > Forward")
  forward(backend, state, input_blobs)
  expected_output = lrn_forward(input, state, op_dim)
  got_output = similar(input)
  copy!(got_output, state.blobs[1])
  @test all(abs.(got_output - expected_output) .< eps)

  println("    > Backward")
  top_diff = rand(T, size(input))
  copy!(state.blobs_diff[1], top_diff)
  backward(backend, state, input_blobs, diff_blobs)
  got_grad = zeros(T, size(input))
  copy!(got_grad, diff_blobs[1])
  expected_grad = lrn_backward(input, top_diff, state, op_dim)
  @test all(abs.(got_grad - expected_grad) .< eps)

  shutdown(backend, state)
end

function lrn_forward_across_channel{T}(input::Array{T}, state, op_dim)
  output = similar(input)
  pre_dim, chann_dim, post_dim = split_dims(input, op_dim)
  pre_pad = div(state.layer.kernel-1,2)
  post_pad = state.layer.kernel - pre_pad - 1

  canonical_input = reshape(input, (pre_dim, chann_dim, post_dim))
  canonical_output = reshape(output, (pre_dim, chann_dim, post_dim))

  for n = 1:post_dim
    for c = 1:chann_dim
      cstart = c-pre_pad
      cend   = min(c + post_pad, chann_dim)
      cstart = max(1, cstart)

      tmp = canonical_input[:,cstart:cend,n].^2 * (state.layer.scale / state.layer.kernel)
      tmp = (sum(tmp, 2) + state.layer.shift) .^ state.layer.power
      canonical_output[:,c,n] = canonical_input[:,c,n] ./ tmp
    end
  end

  return output
end
function lrn_forward_within_channel{T}(input::Array{T}, state)
  output = similar(input)
  width, height, channels, num = size(input)
  pooled_width = width; pooled_height = height
  kernel_size = state.layer.kernel^2
  pre_pad = div(state.layer.kernel-1,2)

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = ph - pre_pad
          wstart = pw - pre_pad
          hend = min(hstart + state.layer.kernel - 1, height)
          wend = min(wstart + state.layer.kernel - 1, width)
          hstart = max(1, hstart)
          wstart = max(1, wstart)

          tmp = (input[wstart:wend,hstart:hend,c,n]).^2 * state.layer.scale / kernel_size
          tmp = (sum(tmp) + state.layer.shift) .^ state.layer.power
          output[pw,ph,c,n] = input[pw,ph,c,n] / tmp
        end
      end
    end
  end

  return output
end
function lrn_forward{T}(input::Array{T}, state, op_dim)
  if isa(state.layer.mode, LRNMode.AcrossChannel)
    lrn_forward_across_channel(input, state, op_dim)
  elseif isa(state.layer.mode, LRNMode.WithinChannel)
    lrn_forward_within_channel(input, state)
  else
    error("Unknown LRN-mode $(state.layer.mode)")
  end
end

function lrn_backward_across_channel{T}(input::Array{T}, top_diff::Array{T}, state, op_dim)
  output = zeros(T, size(input))
  pre_dim, chann_dim, post_dim = split_dims(input, op_dim)
  pre_pad = div(state.layer.kernel-1,2)
  post_pad = state.layer.kernel - pre_pad - 1

  canonical_input = reshape(input, (pre_dim, chann_dim, post_dim))
  canonical_output = reshape(output, (pre_dim, chann_dim, post_dim))
  canonical_diff = reshape(top_diff, (pre_dim, chann_dim, post_dim))

  for n = 1:post_dim
    for c = 1:chann_dim
      cstart = c-pre_pad
      cend   = min(c + post_pad, chann_dim)
      cstart = max(1, cstart)

      tmp = canonical_input[:,cstart:cend,n].^2 * (state.layer.scale / state.layer.kernel)
      tmp = (sum(tmp, 2) + state.layer.shift)

      canonical_output[:,c,n] += tmp .^ (-state.layer.power) .* canonical_diff[:,c,n]

      tmp = -state.layer.power * tmp .^ (-state.layer.power - 1)
      tmp = 2 * state.layer.scale / state.layer.kernel * tmp
      canonical_output[:,cstart:cend,n] += tmp .* canonical_input[:,cstart:cend,n] .*
          canonical_input[:,c,n] .* canonical_diff[:,c,n]
    end
  end

  return output
end
function lrn_backward_within_channel{T}(input::Array{T}, top_diff::Array{T}, state)
  output = zeros(T, size(input))
  width, height, channels, num = size(input)
  pooled_width = width; pooled_height = height
  kernel_size = state.layer.kernel^2
  pre_pad = div(state.layer.kernel-1,2)

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = ph - pre_pad
          wstart = pw - pre_pad
          hend = min(hstart + state.layer.kernel - 1, height)
          wend = min(wstart + state.layer.kernel - 1, width)
          hstart = max(1, hstart)
          wstart = max(1, wstart)

          tmp = (input[wstart:wend,hstart:hend,c,n]).^2 * state.layer.scale / kernel_size
          tmp = (sum(tmp) + state.layer.shift)
          output[pw,ph,c,n] += tmp .^ (-state.layer.power) * top_diff[pw,ph,c,n]

          tmp = -state.layer.power * tmp .^ (-state.layer.power-1)
          tmp = 2 * state.layer.scale / kernel_size * tmp
          output[wstart:wend, hstart:hend, c, n] +=
              tmp * input[wstart:wend, hstart:hend, c, n] * input[pw,ph,c,n] * top_diff[pw,ph,c,n]
        end
      end
    end
  end

  return output
end

function lrn_backward{T}(input::Array{T}, top_diff::Array{T}, state, op_dim)
  if isa(state.layer.mode, LRNMode.AcrossChannel)
    lrn_backward_across_channel(input, top_diff, state, op_dim)
  elseif isa(state.layer.mode, LRNMode.WithinChannel)
    lrn_backward_within_channel(input, top_diff, state)
  else
    error("Unknown LRN-mode $(state.layer.mode)")
  end
end

function test_lrn_layer(backend::Backend, mode::LRNModeType, T, eps)
  test_lrn_layer(backend, mode, 4, T, eps)
end

function test_lrn_layer(backend::Backend, T, eps)
  test_lrn_layer(backend, LRNMode.AcrossChannel(), T, eps)
  test_lrn_layer(backend, LRNMode.WithinChannel(), T, eps)
end

function test_lrn_layer(backend::Backend)
  test_lrn_layer(backend, Float32, 1e-3)
  test_lrn_layer(backend, Float64, 1e-9)
end

if test_cpu
  test_lrn_layer(backend_cpu)
end
if test_gpu
  test_lrn_layer(backend_gpu)
end
