@defstruct ConvolutionLayer TrainableLayer (
  name :: String = "convolution",
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
  (kernel :: NTuple{2,Int} = (1,1), length(kernel)==2 && all([kernel...] .> 0)),
  (stride :: NTuple{2,Int} = (1,1), length(stride)==2 && all([stride...] .> 0)),
  (pad :: NTuple{2, Int} = (0,0), length(pad)==2 && all([pad...] .>= 0)),
  (n_filter :: Int = 1, n_filter > 0),
  (n_group :: Int = 1, n_group > 0),
  neuron :: ActivationFunction = Neurons.Identity(),
  filter_init :: Initializer = XavierInitializer(),
  bias_init :: Initializer = ConstantInitializer(0),
  filter_regu :: Regularizer = L2Regu(1),
  bias_regu :: Regularizer = NoRegu(),
  filter_lr :: FloatingPoint = 1.0,
  bias_lr :: FloatingPoint = 2.0,
)

type CPUConvState
  col_buffer      :: Blob
  M               :: Int
  N               :: Int
  K               :: Int

  # a all-1 vector used in gemm to help bias calculation
  bias_multiplier :: Blob
  img_buffer      :: Array
end

function setup_etc(sys::System{CPUBackend}, layer::ConvolutionLayer, dtype, width, height,
    channels, batch_size, width_out, height_out, inputs)

  if layer.kernel[1] == 1 && layer.kernel[2] == 1 &&
     layer.stride[1] == 1 && layer.stride[2] == 1 &&
     layer.pad[1] == 0 && layer.pad[2] == 0
    col_buffer = NullBlob()
  else
    col_buffer = CPUBlob(Array(dtype, width_out, height_out, channels*prod(layer.kernel), 1))
  end
  M = height_out * width_out
  N = div(layer.n_filter, layer.n_group)
  K = div(channels * layer.kernel[1] * layer.kernel[2], layer.n_group)
  bias_multiplier = CPUBlob(dtype, M)
  fill!(bias_multiplier, convert(dtype,1))
  img_buffer = Array(dtype, width, height, channels)
  etc = CPUConvState(col_buffer, M, N, K, bias_multiplier, img_buffer)
  return etc
end

type ConvolutionLayerState <: LayerState
  layer      :: ConvolutionLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  filter  :: Blob
  ∇filter :: Blob
  bias    :: Blob
  ∇bias   :: Blob

  ConvolutionLayerState(sys::System, layer::ConvolutionLayer, shared_state, inputs::Vector{Blob}) = begin
    channels = get_chann(inputs[1])
    @assert channels % layer.n_group == 0
    @assert layer.n_filter % layer.n_group == 0

    batch_size = get_num(inputs[1])
    height = get_height(inputs[1])
    width  = get_width(inputs[1])
    width_out  = div(width  + 2*layer.pad[1]-layer.kernel[1], layer.stride[1]) + 1
    height_out = div(height + 2*layer.pad[2]-layer.kernel[2], layer.stride[2]) + 1

    dtype = eltype(inputs[1])

    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))

    for i = 1:length(inputs)
      blobs[i] = make_blob(sys.backend, dtype, width_out, height_out, layer.n_filter, batch_size)
      blobs_diff[i] = make_blob(sys.backend, dtype, width_out, height_out, layer.n_filter, batch_size)
    end

    if isa(shared_state, ConvolutionLayerState)
      @assert size(shared_state.filter) == tuple(layer.kernel...,div(channels,layer.n_group),layer.n_filter)
      @assert eltype(shared_state.filter) == dtype
      @debug("Sharing filters and bias with an existing ConvolutionLayer")

      filter = shared_state.filter
      ∇filter = shared_state.∇filter
      bias = shared_state.bias
      ∇bias = shared_state.bias
    else
      filter = make_blob(sys.backend, dtype, layer.kernel[1], layer.kernel[2],
          div(channels,layer.n_group), layer.n_filter)
      ∇filter = make_blob(sys.backend, dtype, layer.kernel[1], layer.kernel[2],
          div(channels,layer.n_group), layer.n_filter)
      bias = make_blob(sys.backend, dtype, layer.n_filter)
      ∇bias = make_blob(sys.backend, dtype, layer.n_filter)
    end

    etc = setup_etc(sys, layer, dtype, width, height, channels, batch_size, width_out, height_out, inputs)

    parameters = [Parameter(filter, ∇filter, layer.filter_init, layer.filter_regu, layer.filter_lr),
                  Parameter(bias, ∇bias, layer.bias_init, layer.bias_regu, layer.bias_lr)]

    state = new(layer, blobs, blobs_diff, parameters)
    state.filter = filter
    state.∇filter = ∇filter
    state.bias = bias
    state.∇bias = ∇bias

    state.height_out = height_out
    state.width_out = width_out

    state.etc = etc

    return state
  end

  # Auxiliary variables
  height_out :: Int
  width_out  :: Int

  etc        :: Any # whatever status a computation backend needs to maintain
end

function setup(sys::System, layer::ConvolutionLayer, shared_state, inputs::Vector{Blob}, diffs::Vector{Blob})
  return ConvolutionLayerState(sys, layer, shared_state, inputs)
end

function forward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    erase!(output)
    dtype = eltype(input)
    width, height, channels, num = size(input)
    img_offset = width*height*channels * sizeof(dtype)
    weight_offset = state.etc.N * state.etc.K * sizeof(dtype)
    col_offset = state.etc.M * state.etc.K * sizeof(dtype)
    top_offset = state.etc.M * state.etc.N * sizeof(dtype)
    top_img_offset = state.height_out * state.width_out * state.layer.n_filter * sizeof(dtype)

    for n = 1:num
      if isa(state.etc.col_buffer, NullBlob)
        col_buffer = convert(Ptr{dtype}, input.data) + img_offset * (n-1)
      else
        col_buffer = state.etc.col_buffer.data
        im2col(input.data, n, col_buffer,
            width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
        col_buffer = convert(Ptr{dtype}, col_buffer)
      end

      output_ptr = convert(Ptr{dtype}, output.data) + top_img_offset * (n-1)
      for g = 1:state.layer.n_group
        RawBLAS.gemm!('N', 'N', state.etc.M, state.etc.N, state.etc.K, convert(dtype, 1),
            col_buffer + col_offset * (g-1),
            convert(Ptr{dtype}, state.filter.data) + weight_offset * (g-1),
            convert(dtype, 0), output_ptr + top_offset * (g-1))
      end
      RawBLAS.gemm!('N', 'N', state.etc.M, state.layer.n_filter, 1, convert(dtype, 1),
          state.etc.bias_multiplier.data, state.bias.data, convert(dtype, 1), output_ptr)
    end
  end
end

function backward(sys::System{CPUBackend}, state::ConvolutionLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  erase!(state.∇filter)
  erase!(state.∇bias)
  for i = 1:length(inputs)
    input = inputs[i]
    top_diff = state.blobs_diff[i]
    dtype = eltype(input)
    width, height, channels, num = size(input)
    img_offset = width*height*channels * sizeof(dtype)
    weight_offset = state.etc.N * state.etc.K * sizeof(dtype)
    col_offset = state.etc.M * state.etc.K * sizeof(dtype)
    top_offset = state.etc.M * state.etc.N * sizeof(dtype)
    top_img_offset = state.height_out * state.width_out * state.layer.n_filter * sizeof(dtype)

    for n = 1:num
      top_diff_ptr = convert(Ptr{dtype}, top_diff.data) + top_img_offset * (n-1)

      #----------------------------------------------
      # bias gradient
      RawBLAS.gemv!('T', state.etc.M, state.layer.n_filter, convert(dtype, 1), top_diff_ptr,
          state.etc.bias_multiplier.data, convert(dtype, 1), state.∇bias.data)

      #----------------------------------------------
      # filter gradient
      if isa(state.etc.col_buffer, NullBlob)
        col_buffer = convert(Ptr{dtype}, input.data) + img_offset * (n-1)
      else
        col_buffer = state.etc.col_buffer.data
        im2col(input.data, n, col_buffer,
            width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
        col_buffer = convert(Ptr{dtype}, col_buffer)
      end
      for g = 1:state.layer.n_group
        RawBLAS.gemm!('T', 'N', state.etc.K, state.etc.N, state.etc.M, convert(dtype, 1),
            col_buffer + col_offset * (g-1),
            top_diff_ptr + top_offset * (g-1), convert(dtype, 1),
            convert(Ptr{dtype}, state.∇filter.data) + weight_offset * (g-1))
      end
    end

    #----------------------------------------------
    # back propagate gradient
    if isa(diffs[i], CPUBlob)
      diff = diffs[i]
      for n = 1:num
        top_diff_ptr = convert(Ptr{dtype}, top_diff.data) + top_img_offset * (n-1)
        if isa(state.etc.col_buffer, NullBlob)
          col_buffer = convert(Ptr{dtype}, diff.data) + img_offset * (n-1)
        else
          col_buffer = convert(Ptr{dtype}, state.etc.col_buffer.data)
        end

        for g = 1:state.layer.n_group
          RawBLAS.gemm!('N', 'T', state.etc.M, state.etc.K, state.etc.N, convert(dtype, 1),
              top_diff_ptr + top_offset * (g-1),
              convert(Ptr{dtype}, state.filter.data) + weight_offset * (g-1),
              convert(dtype, 0), col_buffer + col_offset * (g-1))
        end
        if !(isa(state.etc.col_buffer, NullBlob))
           col2im(state.etc.col_buffer.data, diff.data, n, state.etc.img_buffer,
              width, height, channels, state.layer.kernel, state.layer.pad, state.layer.stride)
        end
      end
    end
  end
end

