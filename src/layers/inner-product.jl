@defstruct InnerProductLayer TrainableLayer (
  name :: String = "inner-product",
  (tops :: Vector{Symbol} = Symbol[], length(tops) >= 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
  (output_dim :: Int = 0, output_dim > 0),
  weight_init :: Initializer = XavierInitializer(),
  bias_init :: Initializer = ConstantInitializer(0),
  weight_regu :: Regularizer = L2Regu(1),
  bias_regu :: Regularizer = NoRegu(),
  weight_lr :: FloatingPoint = 1.0,
  bias_lr :: FloatingPoint = 2.0,
  neuron :: ActivationFunction = Neurons.Identity()
)

type InnerProductLayerState <: LayerState
  layer      :: InnerProductLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  W  :: Blob
  ∇W :: Blob
  b  :: Blob
  ∇b :: Blob

  # a all-1 vector used in gemm to help bias calculation
  bias_multiplier :: Blob

  InnerProductLayerState(sys::System, layer::InnerProductLayer, shared_state, inputs::Vector{Blob}) = begin
    dims = size(inputs[1])
    nums = dims[4]
    fea_dim = dims[1:3]
    fea_size = prod(fea_dim)
    out_dim = layer.output_dim

    data_type = eltype(inputs[1])
    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))

    for i = 1:length(inputs)
      blobs[i] = make_blob(sys.backend, data_type, out_dim, nums)
      blobs_diff[i] = make_blob(sys.backend, data_type, out_dim, nums)
    end

    state = new(layer, blobs, blobs_diff)

    if isa(shared_state, InnerProductLayerState)
      @assert size(shared_state.W) == (1, 1, fea_size, out_dim)
      @assert eltype(shared_state.W) == data_type
      @debug("Sharing weights and bias with an existing InnerProductLayer")

      state.W  = shared_state.W
      state.∇W = shared_state.∇W
      state.b  = shared_state.b
      state.∇b = shared_state.∇b
    else
      state.W  = make_blob(sys.backend, data_type, fea_size, out_dim)
      state.∇W = make_blob(sys.backend, data_type, fea_size, out_dim)
      state.b  = make_blob(sys.backend, data_type, out_dim)
      state.∇b = make_blob(sys.backend, data_type, out_dim)
    end
    state.bias_multiplier = make_blob(sys.backend, data_type, nums)
    fill!(state.bias_multiplier, 1)

    state.parameters = [Parameter(state.W, state.∇W, layer.weight_init, layer.weight_regu, layer.weight_lr),
                        Parameter(state.b, state.∇b, layer.bias_init, layer.bias_regu, layer.bias_lr)]

    return state
  end
end

function setup(sys::System, layer::InnerProductLayer, shared_state, inputs::Vector{Blob})
  state = InnerProductLayerState(sys, layer, shared_state, inputs)
  return state
end

function forward(sys::System{CPUBackend}, state::InnerProductLayerState, inputs::Vector{Blob})
  M = size(state.W, 4)   # target dim
  N = size(inputs[1], 4) # batch size
  K = size(state.W, 3)   # source dim
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    # output = W^T * X
    BLAS.gemm!('T', 'N', convert(dtype, 1), reshape(state.W.data, (K,M)),
                reshape(input.data, (K,N)), convert(dtype, 0), reshape(output.data, (M,N)))
    # output += bias
    BLAS.gemm!('N', 'N', convert(dtype, 1), reshape(state.b.data, (M,1)),
                reshape(state.bias_multiplier.data, (1,N)), convert(dtype, 1), reshape(output.data, (M,N)))
  end
end

function backward(sys::System{CPUBackend}, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  target_dim = size(state.W, 4)
  source_dim = size(state.W, 3)
  batch_size = size(inputs[1], 4)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = convert(data_type, 0)

  for i = 1:length(inputs)
    # ∂f/∂W = input * [∂f/∂o]^T
    input = inputs[i]
    ∂f_∂o = state.blobs_diff[i]
    BLAS.gemm!('N', 'T', convert(data_type, 1), reshape(input.data, (source_dim, batch_size)),
               reshape(∂f_∂o.data, (target_dim, batch_size)), zero_and_then_one,
               reshape(state.∇W.data, (source_dim, target_dim)))

    # ∂f/∂b = sum(∂f/∂o, 2)
    BLAS.gemm!('N', 'N', convert(data_type, 1), reshape(∂f_∂o.data, (target_dim, batch_size)),
               reshape(state.bias_multiplier.data, (batch_size, 1)), zero_and_then_one,
               reshape(state.∇b.data, (target_dim, 1)))

    zero_and_then_one = convert(data_type, 1)

    # if back propagate down
    if isa(diffs[i], CPUBlob)
      # ∂f/∂x = W * [∂f/∂o]
      BLAS.gemm!('N', 'N', convert(data_type, 1), reshape(state.W.data, (source_dim, target_dim)),
                 reshape(∂f_∂o.data, (target_dim, batch_size)), convert(data_type, 0),
                 reshape(diffs[i].data, (source_dim, batch_size)))
    end
  end
end


