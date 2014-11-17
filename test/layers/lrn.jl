function test_lrn_layer(sys::System, mode::LRNModeType)
  println("-- Testing LRN($(typeof(mode))) on $(typeof(sys.backend))...")
  println("    > Setup")

  eps = 1e-10
  input = rand(7,8,9,10)
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
