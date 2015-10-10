@defstruct TiedInnerProductLayer Layer (
  (name :: AbstractString = "", !isempty(name)),
  param_key :: AbstractString = "",
  (tied_param_key :: AbstractString = "", !isempty(tied_param_key)),
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
  bias_init :: Initializer = ConstantInitializer(0),
  bias_regu :: Regularizer = NoRegu(),
  bias_cons :: Constraint = NoCons(),
  bias_lr :: AbstractFloat = 2.0,
  neuron :: ActivationFunction = Neurons.Identity()
)
@characterize_layer(TiedInnerProductLayer,
  can_do_bp  => true,
  has_param  => true,
  has_neuron => true
)

type TiedInnerProductLayerState <: LayerState
  layer      :: TiedInnerProductLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}

  parameters :: Vector{Parameter}

  W  :: Blob # reference to weights of tied layer, no learning for W here

  b  :: Blob
  ∇b :: Blob

  # a all-1 vector used in gemm to help bias calculation
  bias_multipliers :: Vector{Blob}

  frozen :: Bool

  TiedInnerProductLayerState(backend::Backend, layer::TiedInnerProductLayer, shared_params, inputs::Vector{Blob}) = begin
    fea_size  = get_fea_size(inputs[1])
    data_type = eltype(inputs[1])
    for i = 2:length(inputs)
      @assert get_fea_size(inputs[i]) == fea_size
      @assert eltype(inputs[i]) == data_type
    end

    tied_params = registry_get(backend, layer.tied_param_key)
    @assert length(tied_params) == 2
    @assert tied_params[1].name == "weight"
    W = tied_params[1].blob
    @assert size(W, 2) == fea_size
    out_dim = size(W, 1)

    blobs = Array(Blob, length(inputs))
    blobs_diff = Array(Blob, length(inputs))
    bias_multipliers = Array(Blob, length(inputs))

    for i = 1:length(inputs)
      nums = get_num(inputs[i])
      blobs[i] = make_blob(backend, data_type, out_dim, nums)
      blobs_diff[i] = make_blob(backend, data_type, out_dim, nums)
      bias_multipliers[i] = make_blob(backend, ones(data_type, 1, nums))
    end

    state = new(layer, blobs, blobs_diff)
    state.bias_multipliers = bias_multipliers

    if shared_params != Void()
      @assert length(shared_params) == 1
      @assert shared_params[1].name == "bias"
      @assert size(shared_params[1].blob) == (out_dim, 1)
      @debug("TiedInnerProductLayer($(layer.name)): sharing bias")

      params = [share_parameter(backend, shared_params[1])]
    else
      params = [make_parameter(backend, "bias", data_type, (out_dim,1),
          layer.bias_init, layer.bias_regu, layer.bias_cons, layer.bias_lr)]
    end

    state.W = W
    state.b = params[1].blob
    state.∇b = params[1].gradient
    state.parameters = params
    state.frozen = false

    return state
  end
end

function freeze!(state::TiedInnerProductLayerState)
  state.frozen = true
end
function unfreeze!(state::TiedInnerProductLayerState)
  state.frozen = false
end
function is_frozen(state::TiedInnerProductLayerState)
  state.frozen
end

function setup(backend::Backend, layer::TiedInnerProductLayer, shared_state, inputs::Vector{Blob}, diffs::Vector{Blob})
  TiedInnerProductLayerState(backend, layer, shared_state, inputs)
end

function setup(backend::Backend, layer::TiedInnerProductLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  setup(backend, layer, Void(), inputs, diffs)
end

function shutdown(backend::Backend, state::TiedInnerProductLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  map(destroy, state.bias_multipliers)
  map(destroy, state.parameters)
end

function forward(backend::CPUBackend, state::TiedInnerProductLayerState, inputs::Vector{Blob})
  # note state.W is literally from the tied layer, we should use its transpose
  recon_dim  = size(state.W, 1)
  hidden_dim = size(state.W, 2)

  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    N = get_num(input)
    output = state.blobs[i]
    # output = (W^T)^T * X
    BLAS.gemm!('N', 'N', one(dtype), state.W.data,
                reshape(input.data, (hidden_dim,N)), zero(dtype), output.data)
    # output += bias
    BLAS.gemm!('N', 'N', one(dtype), state.b.data,
                state.bias_multipliers[i].data, one(dtype), output.data)
  end
end

function backward(backend::CPUBackend, state::TiedInnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  recon_dim = size(state.W, 1)
  hidden_dim = size(state.W, 2)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = zero(data_type)

  for i = 1:length(inputs)
    input = inputs[i]
    batch_size = get_num(input)
    ∂f_∂o = state.blobs_diff[i]

    if !state.frozen
      # ∂f/∂b = sum(∂f/∂o, 2)
      BLAS.gemm!('N', 'N', one(data_type), ∂f_∂o.data,
                 reshape(state.bias_multipliers[i].data, (batch_size, 1)), zero_and_then_one,
                 state.∇b.data)
    end

    zero_and_then_one = one(data_type)

    # if back propagate down
    if isa(diffs[i], CPUBlob)
      # ∂f/∂x = W^T * [∂f/∂o]
      BLAS.gemm!('T', 'N', one(data_type), state.W.data,
                 ∂f_∂o.data, zero(data_type),
                 reshape(diffs[i].data, (hidden_dim, batch_size)))
    end
  end
end
