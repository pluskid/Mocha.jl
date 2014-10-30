@defstruct ConvolutionLayer CompLayer (
  (kernel :: Vector{Int} = [1,1], length(kernel)==2 && all(kernel .> 0)),
  (stride :: Vector{Int} = [1,1], length(stride)==2 && all(stride .> 0)),
  (pad :: Vector{Int} = [0,0], length(pad)==2 && all(pad .>= 0)),
  (n_filter :: Int = 1, n_filter > 0),
  (n_group :: Int = 1, n_group > 0),
  neuron :: ActivationFunction = Neurons.Identity(),
  (bottoms :: Vector{String} = String[], length(bottoms) > 0),
  (tops :: Vector{String} = String[], length(tops) == length(bottoms)),
  filter_init :: Initializer = ConstantInitializer(0),
  bias_init :: Initializer = ConstantInitializer(0),
  filter_regu :: Regularizer = NoRegu(),
  bias_regu :: Regularizer = NoRegu()
)

type ConvolutionLayerState <: LayerState
  layer      :: ConvolutionLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  filter  :: Blob
  ∇filter :: Blob
  bias    :: Blob
  ∇bias   :: Blob

  ConvolutionLayerState(sys::System, layer::ConvolutionLayer, inputs::Vector{Blob}) = begin
    channels = size(inputs[1],2)
    @assert channels % layer.n_group == 0
    @assert layer.n_filter % layer.n_group == 0

    height, width = size(inputs[1])[3:4]
    height_out = int((height + 2*layer.pad[1]-layer.kernel[1]) / layer.stride[1]) + 1
    width_out  = int((width  + 2*layer.pad[2]-layer.kernel[2]) / layer.stride[2]) + 1
    is_1x1 = all(layer.kernel .== 1) && all(layer.pad .== 0) && all(layer.stride .== 1)

    dtype = eltype(inputs[1])

    if isa(sys.backend, CPUBackend)
      blobs = Array(Blob, length(inputs))
      blobs_diff = Array(Blob, length(inputs))
      for i = 1:length(inputs)
        blobs[i] = CPUBlob(Array(dtype, size(inputs[i],1), layer.n_filter, height_out, width_out))
        blobs_diff[i] = CPUBlob(similar(blobs[i].data))
      end

      filter = CPUBlob(Array(dtype, layer.n_filter, int(channels/layer.n_group), layer.kernel...))
      ∇filter = CPUBlob(similar(filter.data))
      bias = CPUBlob(Array(dtype, layer.n_filter))
      ∇bias = CPUBlob(similar(bias.data))

      col_buffer = CPUBlob(Array(dtype, 1, channels*prod(layer.kernel), height_out, width_out))
    else
      error("Backend $(sys.backend) not supported")
    end

    parameters = [Parameter(filter, ∇filter, layer.filter_init, layer.filter_regu),
                  Parameter(bias, ∇bias, layer.bias_init, layer.bias_regu)]

    state = new(layer, blobs, blobs_diff, parameters)
    state.filter = filter
    state.∇filter = ∇filter
    state.bias = bias
    state.∇bias = ∇bias

    state.height_out = height_out
    state.width_out = width_out

    return state
  end

  # Auxiliary variables
  height_out :: Int
  width_out  :: Int
end

function setup(sys::System{CPUBackend}, layer::ConvolutionLayer, inputs::Vector{Blob})
  return ConvolutionLayerState(sys, layer, inputs)
end

function forward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob})
  channel, height, width = size(inputs[1])[2:end]
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs[i].data

    for n = 1:size(input,1)
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:state.height_out
              for x = 1:state.width_out
                output[n, (g-1)*o_g+o, y, x] = state.bias.data[(g-1)*o_g+o]

                for p = 1:state.layer.kernel[1]
                  for q = 1:state.layer.kernel[2]
                    in_y = (y-1) * state.layer.stride[1] - state.layer.pad[1] + p
                    in_x = (x-1) * state.layer.stride[2] - state.layer.pad[2] + q
                    if (in_y >= 1 && in_y <= height && in_x >= 1 && in_y <= width)
                      output[n, (g-1)*o_g+o, y, x] += input[n, (g-1)*k_g+k, in_y, in_x] *
                                                      state.filter.data[(g-1)*o_g+o, k, p, q]
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
end

function backward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  channel, height, width = size(inputs[1])[2:end]
  n_group = state.layer.n_group
  o_g = int(state.layer.n_filter / n_group)
  k_g = int(channel / n_group)

  fill!(state.∇filter.data, 0)

  for i = 1:length(inputs)
    input = inputs[i].data
    output = state.blobs_diff[i].data

    diff = diffs[i]
    if !isa(diff, NullBlob)
      fill!(diff.data, 0)
    end

    for n = 1:size(input,1)
      for g = 1:n_group
        for o = 1:o_g
          for k = 1:k_g
            for y = 1:state.height_out
              for x = 1:state.width_out
                state.∇bias.data[(g-1)*o_g+o] = output[n, (g-1)*o_g+o, y, x]

                for p = 1:state.layer.kernel[1]
                  for q = 1:state.layer.kernel[2]
                    in_y = (y-1) * state.layer.stride[1] - state.layer.pad[1] + p
                    in_x = (x-1) * state.layer.stride[2] - state.layer.pad[2] + q
                    if (in_y >= 1 && in_y <= height && in_x >= 1 && in_y <= width)
                      state.∇filter.data[(g-1)*o_g+o,k,p,q] += output[n,(g-1)*o_g+o,y,x] *
                                                               input[n,(g-1)*k_g+k,in_y,in_x]
                      if !isa(diff, NullBlob)
                        diff.data[n,(g-1)*k_g+k,in_y,in_x] += output[n,(g-1)*o_g+o,y,x] *
                                                              state.filter.data[(g-1)*o_g+o,k,p,q]
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
  end
end

