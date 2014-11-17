function test_lrn_layer(sys::System, mode::LRNModeType)
  println("-- Testing LRN($(typeof(mode))) on $(typeof(sys.backend))...")
  println("    > Setup")

  eps = 1e-5
  input = float(int(100*rand(7,8,9,10)))/100
  input_blobs = Blob[make_blob(sys.backend, input)]
  diff_blobs = Blob[make_blob(sys.backend, input)]

  layer = LRNLayer(tops=[:output],bottoms=[:input], mode=mode)
  state = setup(sys, layer, input_blobs, diff_blobs)

  @test size(state.blobs[1]) == size(input)

  println("    > Forward")
  forward(sys, state, input_blobs)
  expected_output = lrn_forward(input, state)
  got_output = similar(input)
  copy!(got_output, state.blobs[1])
  @test all(abs(got_output - expected_output) .< eps)

  println("    > Backward")
  top_diff = float(int(100*rand(size(input))))/100
  copy!(state.blobs_diff[1], top_diff)
  backward(sys, state, input_blobs, diff_blobs)
  got_grad = zeros(size(input))
  copy!(got_grad, diff_blobs[1])
  expected_grad = lrn_backward(input, top_diff, state)
  #println("in $(input[:])")
  #println("top_diff $(top_diff[:])")
  #println("exp $(expected_grad[:])")
  #println("got $(got_grad[:])")
  @test all(abs(got_grad - expected_grad) .< eps)
end

function lrn_forward_across_channel{T}(input::Array{T}, state)
  output = similar(input)
  width, height, channels, num = size(input)
  pre_pad = div(state.layer.kernel-1,2)
  post_pad = state.layer.kernel - pre_pad - 1

  for n = 1:num
    for c = 1:channels
      cstart = c-pre_pad
      cend   = min(c + post_pad, channels)
      cstart = max(1, cstart)

      tmp = input[:,:,cstart:cend,n].^2 * (state.layer.scale / state.layer.kernel)
      tmp = (sum(tmp, 3) + state.layer.shift) .^ state.layer.power
      output[:,:,c,n] = input[:,:,c,n] ./ tmp
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
function lrn_forward{T}(input::Array{T}, state)
  if isa(state.layer.mode, LRNMode.AcrossChannel)
    lrn_forward_across_channel(input, state)
  elseif isa(state.layer.mode, LRNMode.WithinChannel)
    lrn_forward_within_channel(input, state)
  else
    error("Unknown LRN-mode $(state.layer.mode)")
  end
end

function lrn_backward_across_channel{T}(input::Array{T}, top_diff::Array{T}, state)
  output = zeros(size(input))
  width, height, channels, num = size(input)
  pre_pad = div(state.layer.kernel-1,2)
  post_pad = state.layer.kernel - pre_pad - 1

  for n = 1:num
    for c = 1:channels
      cstart = c-pre_pad
      cend   = min(c + post_pad, channels)
      cstart = max(1, cstart)

      tmp = input[:,:,cstart:cend,n].^2 * (state.layer.scale / state.layer.kernel)
      tmp = (sum(tmp, 3) + state.layer.shift)

      output[:,:,c,n] += tmp .^ (-state.layer.power) .* top_diff[:,:,c,n]

      tmp = -state.layer.power * tmp .^ (-state.layer.power - 1)
      tmp = 2 * state.layer.scale / state.layer.kernel * tmp
      output[:,:,cstart:cend,n] += tmp .* input[:,:,cstart:cend,n] .* input[:,:,c,n] .* top_diff[:,:,c,n]
    end
  end

  return output
end
function lrn_backward_within_channel{T}(input::Array{T}, top_diff::Array{T}, state)
  output = zeros(size(input))
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

function lrn_backward{T}(input::Array{T}, top_diff::Array{T}, state)
  if isa(state.layer.mode, LRNMode.AcrossChannel)
    lrn_backward_across_channel(input, top_diff, state)
  elseif isa(state.layer.mode, LRNMode.WithinChannel)
    lrn_backward_within_channel(input, top_diff, state)
  else
    error("Unknown LRN-mode $(state.layer.mode)")
  end
end

function test_lrn_layer(sys::System)
  test_lrn_layer(sys, LRNMode.AcrossChannel())
  test_lrn_layer(sys, LRNMode.WithinChannel())
end

if test_cpu
  test_lrn_layer(sys_cpu)
end
if test_cudnn
  test_lrn_layer(sys_cudnn)
end
