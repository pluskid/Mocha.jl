@defstruct WassersteinLossLayer Layer (
  name :: AbstractString = "wasserstein-loss",
  (M :: Matrix = [], !isempty(M)),
  (lambda :: AbstractFloat = 0, lambda > 0),
  (sinkhorn_iter :: Int = 50, sinkhorn_iter > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2)
)
@characterize_layer(WassersteinLossLayer,
  has_loss  => true,
  can_do_bp => true,
  is_sink   => true,
)

type WassersteinLossLayerState{T} <: LayerState
  layer :: WassersteinLossLayer
  loss  :: T

  K     :: Blob
  KM    :: Blob
  alpha :: Blob
  u     :: Blob
  tmps  :: Vector{Blob}
end

function setup(backend::Backend, layer::WassersteinLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])

  pred  = inputs[1]
  label = inputs[2]
  @assert size(layer.M, 1) == get_fea_size(pred)
  @assert size(layer.M, 2) == get_fea_size(label)

  K = convert(Array{data_type}, exp(-layer.lambda * layer.M))
  KM = K .* convert(Array{data_type}, layer.M)
  alpha  = make_blob(backend, zeros(data_type, get_fea_size(pred), get_num(pred)))
  u = make_blob(backend, zeros(data_type, get_fea_size(pred), get_num(pred)))
  state = WassersteinLossLayerState(layer, zero(data_type),
      make_blob(backend, K), make_blob(backend, KM), alpha, u, Blob[])
  return state
end

function shutdown(backend::Backend, state::WassersteinLossLayerState)
  destroy(state.K)
  destroy(state.KM)
  destroy(state.alpha)
  destroy(state.u)
  map(destroy, state.tmps)
end

function sinkhorn(backend::CPUBackend, state::WassersteinLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  pred_size = get_fea_size(pred)
  pred_num  = get_num(pred)

  # init as uniform distribution
  copy!(state.u, ones(data_type, pred_size, pred_num) / pred_size);
  u = state.u.data

  a = reshape(pred.data, pred_size, pred_num)
  b = reshape(label.data, get_fea_size(label), pred_num)
  K = state.K.data

  for iter = 1:state.layer.sinkhorn_iter
    u = a ./ (K * (b./(u'*K)'))
  end

  # compute objective function
  v = b ./ (K'*u)
  state.loss = sum(u .* ((state.KM.data)*v)) / pred_num

  # compute gradient
  alpha = log(u) / convert(data_type, state.layer.lambda) / pred_num
  copy!(state.alpha, alpha)
end

function forward(backend::Backend, state::WassersteinLossLayerState, inputs::Vector{Blob})
  sinkhorn(backend, state, inputs)
end

function backward(backend::Backend, state::WassersteinLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if !isa(diff, NullBlob)
    copy!(diff, state.alpha)
  end
end
