function test_convolution_layer(backend::Backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, freeze, deconv, T, eps)
  println("-- Testing $(deconv?"De-":"")Convolution(frozen=$freeze) on $(typeof(backend)){$T} filter=$((filter_w,filter_h))...")
  println("    > Setup")
  input_w = 16
  input_h = 10
  input_chann = 6
  input_num = 24
  n_filter = 12

  input_dims = (input_w, input_h, input_chann, input_num)
  if deconv
    filter_dims = (filter_w, filter_h, div(n_filter,n_group), input_chann)
  else
    filter_dims = (filter_w, filter_h, div(input_chann,n_group), n_filter)
  end
  bias_dims = (1, 1, n_filter, 1)

  layer = ConvolutionLayer(name="conv", kernel=(filter_w, filter_h), stride=(stride_w, stride_h),
      pad=(pad_w,pad_h), n_filter=n_filter, n_group=n_group, deconv=deconv,
      tops=Array(Symbol,n_input), bottoms=Array(Symbol,n_input))

  # convolution layer requires each input blob to be the same shape
  input = [simplerand(T, input_dims) for i = 1:n_input]
  inputs = Blob[make_blob(backend, x) for x in input]
  data_diffs = Blob[make_blob(backend, x) for x in input]

  state = setup(backend, layer, inputs, data_diffs)

  println("    > Forward")
  filter = simplerand(T, filter_dims)
  copy!(state.filter, filter)
  bias = simplerand(T, bias_dims)
  copy!(state.bias, bias)

  forward(backend, state, inputs)
  for i = 1:n_input
    expected_output = convolution_forward(state, filter, bias, input[i])
    @test size(expected_output) == size(state.blobs[i])
    got_output = to_array(state.blobs[i])
    @test all(-eps .< expected_output - got_output .< eps)
  end

  println("    > Backward")
  top_diff = [simplerand(T, size(state.blobs[1])) for i = 1:n_input]
  for i = 1:n_input
    copy!(state.blobs_diff[i], top_diff[i])
  end

  if freeze
    freeze!(state)
  end
  backward(backend, state, inputs, data_diffs)

  grad_filter_exp = zeros(T, filter_dims)
  grad_bias_exp = zeros(T, bias_dims)
  for i = 1:n_input
    gradients_expected = convolution_backward(state, filter, bias, input[i], top_diff[i])
    gradients_got = similar(gradients_expected[3])
    copy!(gradients_got, data_diffs[i])

    @test all(-eps .< gradients_got - gradients_expected[3] .< eps)

    grad_filter_exp += gradients_expected[1]
    grad_bias_exp += gradients_expected[2]
  end
  grad_filter_got = similar(grad_filter_exp)
  grad_bias_got = similar(grad_bias_exp)
  copy!(grad_filter_got, state.∇filter)
  copy!(grad_bias_got, state.∇bias)

  is_grad_filter_match = all(abs(grad_filter_exp - grad_filter_got) .< eps)
  is_grad_bias_match   = all(abs(grad_bias_exp - grad_bias_got) .< eps)

  if freeze
    # when frozen, the gradients are not computed, so the got value
    # should be random un-initialized values, so should not match (with
    # very high probability)
    @test !is_grad_bias_match
    @test !is_grad_filter_match
  else
    @test is_grad_filter_match
    @test is_grad_bias_match
  end

  shutdown(backend, state)
end

# naive implementation of convolution forward, used to check the correctness
function convolution_forward(state, filter::Array, bias::Array, input::Array)
  width, height, channel, num = size(input)
  output = zeros(eltype(input), size(state.blobs[1]))

  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  if state.layer.deconv
    ## Deconvolution
    for n = 1:num
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:height
              for x = 1:width
                for p = 1:state.layer.kernel[2]
                  for q = 1:state.layer.kernel[1]
                    out_y = (y-1) * state.layer.stride[2] - state.layer.pad[2] + p
                    out_x = (x-1) * state.layer.stride[1] - state.layer.pad[1] + q
                    if (out_y >= 1 && out_y <= state.height_out && out_x >= 1 && out_x <= state.width_out)
                      output[out_x,out_y,(g-1)*o_g+o,n] += input[x,y,(g-1)*k_g+k,n] *
                          filter[q,p,o,(g-1)*k_g+k]
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  else
    ## Ordinary Convolution
    for n = 1:num
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:state.height_out
              for x = 1:state.width_out
                for p = 1:state.layer.kernel[2]
                  for q = 1:state.layer.kernel[1]
                    in_y = (y-1) * state.layer.stride[2] - state.layer.pad[2] + p
                    in_x = (x-1) * state.layer.stride[1] - state.layer.pad[1] + q
                    if (in_y >= 1 && in_y <= height && in_x >= 1 && in_x <= width)
                      output[x, y, (g-1)*o_g+o, n] += input[in_x, in_y, (g-1)*k_g+k, n] *
                          filter[q, p, k, (g-1)*o_g+o]
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  # add bias
  for n = 1:num
    for o = 1:state.layer.n_filter
      for y = 1:state.height_out
        for x = 1:state.width_out
          output[x, y, o, n] += bias[o]
        end
      end
    end
  end

  return output
end

# naive implementation of convolution backward, used to check correctness
function convolution_backward(state, filter::Array, bias::Array, input::Array, top_diff::Array)
  ∇filter = zeros(eltype(filter), size(filter))
  ∇bias   = zeros(eltype(bias), size(bias))
  ∇input  = zeros(eltype(input), size(input))

  width, height, channels, num = size(input)
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channels / n_group)

  # ∇bias
  for n = 1:num
    for o = 1:state.layer.n_filter
      for y = 1:state.height_out
        for x = 1:state.width_out
          ∇bias[o] += top_diff[x, y, o, n]
        end
      end
    end
  end

  # ∇filter and ∇input
  if state.layer.deconv
    # Deconvolution
    for n = 1:num
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:height
              for x = 1:width
                for p = 1:state.layer.kernel[2]
                  for q = 1:state.layer.kernel[1]
                    out_y = (y-1) * state.layer.stride[2] - state.layer.pad[2] + p
                    out_x = (x-1) * state.layer.stride[1] - state.layer.pad[1] + q
                    if (out_y >= 1 && out_y <= state.height_out && out_x >= 1 && out_x <= state.width_out)
                      ∇filter[q,p,o,(g-1)*k_g+k] += top_diff[out_x,out_y,(g-1)*o_g+o,n] * input[x,y,(g-1)*k_g+k,n]
                      ∇input[x,y,(g-1)*k_g+k,n] += top_diff[out_x,out_y,(g-1)*o_g+o,n] * filter[q,p,o,(g-1)*k_g+k]
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  else
    # Ordinary convolution
    for n = 1:num
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:state.height_out
              for x = 1:state.width_out
                for p = 1:state.layer.kernel[2]
                  for q = 1:state.layer.kernel[1]
                    in_y = (y-1) * state.layer.stride[2] - state.layer.pad[2] + p
                    in_x = (x-1) * state.layer.stride[1] - state.layer.pad[1] + q
                    if (in_y >= 1 && in_y <= height && in_x >= 1 && in_x <= width)
                      ∇filter[q,p,k,(g-1)*o_g+o] += top_diff[x,y,(g-1)*o_g+o,n] * input[in_x,in_y,(g-1)*k_g+k,n]
                      ∇input[in_x,in_y,(g-1)*k_g+k,n] += top_diff[x,y,(g-1)*o_g+o,n] * filter[q,p,k,(g-1)*o_g+o]
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  return (∇filter, ∇bias, ∇input)
end

function test_convolution_layer(backend::Backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, freeze, T, eps)
  test_convolution_layer(backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, freeze, false, T, eps)
  test_convolution_layer(backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, freeze, true, T, eps)
end
function test_convolution_layer(backend::Backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, T, eps)
  test_convolution_layer(backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, true, T, eps)
  test_convolution_layer(backend, n_group, filter_w, filter_h, pad_w, pad_h, stride_w, stride_h, n_input, false, T, eps)
end
function test_convolution_layer(backend::Backend, n_input, T, eps)
  test_convolution_layer(backend, 1, 1, 1, 0, 0, 1, 1, n_input, T, eps)
  test_convolution_layer(backend, 1, 3, 4, 2, 2, 1, 2, n_input, T, eps)
  #test_convolution_layer(backend, 2, 3, 4, 2, 2, 1, 2, n_input, T, eps)
end

function test_convolution_layer(backend::Backend)
  test_convolution_layer(backend, 3, Float64, 1e-10)
  test_convolution_layer(backend, 3, Float32, 1e-1) # Float32 is sooo inaccurate?
end
if test_cpu
  test_convolution_layer(backend_cpu)
end
if test_gpu
  test_convolution_layer(backend_gpu)
end
