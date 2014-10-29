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
  bias_regu :: Regularizer = NoRegu(),
  neuron :: ActivationFunction = Neurons.Identity()
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
    channels = size(input[1],2)
    @assert channels % layer.group == 0
    @assert layer.n_filter % layer.group == 0

    height, width = size(inputs[1])[3:4]
    height_out = (height + 2*layer.pad[1]-layer.kernel[1]) / layer.stride[1] + 1
    width_out  = (width  + 2*layer.pad[2]-layer.kernel[2]) / layer.stride[2] + 1
    is_1x1 = all(layer.kernel .== 1) && all(layer.pad .== 0) && all(layer.stride .== 1)

    dtype = eltype(input[1])

    if isa(sys.backend, CPU)
      blobs = Array(Blob, length(inputs))
      blobs_diff = Array(Blob, length(inputs))
      for i = 1:length(inputs)
        blobs[i] = CPUBlob(Array(dtype, size(inputs[i],1), layer.n_filter, height_out, width_out))
        blobs_diff[i] = CPUBlob(similar(blobs[i].data))
      end

      filter = CPUBlob(Array(dtype, layer.n_filter, channels/layer.group, layer.kernel...))
      ∇filter = CPUBlob(similar(filter.data))
      bias = CPUBlob(Array(dtype, layer.n_filter, 1))
      ∇bias = CPUBlob(similar(bias.data))

      col_buffer = CPUBlob(Array(dtype, 1, channels*prod(layer.kernel), height_out, width_out))
    else
      error("Backend $(sys.backend) not supported")
    end

    parameters = [Parameter(filter, ∇filter, filter_init, filter_regu),
                  Parameter(bias, ∇bias, bias_init, bias_regu)]

    state = new(layer, blobs, blobs_diff, parameters)
    state.filter = filter
    state.∇filter = ∇filter
    state.bias = bias
    state.∇bias = ∇bias
    state.col_buffer = col_buffer

    state.height_out = height_out
    state.width_out = width_out
    state.is_1x1 = is_1x1

    return state
  end

  # Auxiliary variables
  col_buffer :: Blob
  height_out :: Int
  width_out  :: Int
  is_1x1     :: Bool
end

function setup(sys::System{CPU}, layer::ConvolutionLayer, inputs::Vector{Blob})
  return ConvolutionLayerState(sys, layer, inputs)
end

function forward(sys::System{CPU}, state::ConvolutionLayerState, inputs::Vector{Blob})
  channel, height, width = size(inputs[1])[2:end]

  M = state.layer.n_filter / state.layer.group
  K = channel * prod(state.layer.kernel) / group
  N = state.height_out * state.width_out

  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    for n = 1:size(input, 1)
      # caffe-style convolution by im2col
      if (state.is_1x1)
        state.col_buffer.data[:] = input.data[n,:,:,:]
      else
        im2col(sub(input.data, n, 1:channel, 1:height, 1:width), state.col_buffer.data,
               state.layer.kernel, state.layer.pad, state.layer.stride)
      end

      for g = 1:state.layer.group
        idx_g = (g-1)*M+1:g*M
        output.data[n, idx_g, :, :] = reshape(state.filter.data[idx_g,:,:,:], M, K) *
                                      reshape(state.col_buffer[1,idx_g,:,:], K, N) .+
                                      state.bias[idx_g,1]
      end
    end
  end
end

function backward(sys::System{CPU}, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})

end
