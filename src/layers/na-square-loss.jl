############################################################
# Square Loss that is aware of an NA value
# for which no error is backpropagated
#
# L(\hat{y},y) = 1/2N \sum_{i=1}^N (\hat{y}_i - y_i)^2
############################################################
@defstruct NaSquareLossLayer Layer (
  name :: AbstractString = "na-square-loss",
  NAvalue :: AbstractFloat = -9999.0,
  (weight :: AbstractFloat = 1.0, weight >= 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(NaSquareLossLayer,
  has_loss  => true,
  can_do_bp => true,
  is_sink   => true,
  has_stats => true,
)

type NaSquareLossLayerState{T} <: LayerState
  layer      :: NaSquareLossLayer
  loss       :: T

  loss_accum :: T
  n_accum    :: Int

  NAvalue :: T

  # a helper blob used to compute the loss without destroying
  # the pred results passed up
  pred_copy :: Blob
end

function setup(backend::Backend, layer::NaSquareLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])
  pred_copy = make_blob(backend, data_type, size(inputs[1])...)

  state = NaSquareLossLayerState(layer, zero(data_type), zero(data_type), 0, convert(data_type, layer.NAvalue), pred_copy)
  return state
end
function shutdown(backend::Backend, state::NaSquareLossLayerState)
  destroy(state.pred_copy)
end

function reset_statistics(state::NaSquareLossLayerState)
  state.n_accum = 0
  state.loss_accum = zero(typeof(state.loss_accum))
end
function dump_statistics(storage, state::NaSquareLossLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-square-loss", state.loss_accum)

  if show
    loss = @sprintf("%.4f", state.loss_accum)
    @info("  Square-loss (avg over $(state.n_accum)) = $loss")
  end
end

function forward(backend::CPUBackend, state::NaSquareLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  #pred_copy[i] <- pred[i] - label[i]
  copy!(state.pred_copy, pred)
  BLAS.axpy!(n, convert(data_type, -1), label.data, 1, state.pred_copy.data, 1)
  rmna!(label.data, state.pred_copy.data, state.NAvalue)
  # total loss:
  state.loss = state.layer.weight * 0.5/get_num(pred) * BLAS.dot(state.pred_copy.data, state.pred_copy.data)

  # accumulate statistics
  state.loss_accum = (state.loss_accum*state.n_accum + state.loss*get_num(pred)) / (state.n_accum+get_num(pred))
  state.n_accum += get_num(pred)
end

function backward(backend::CPUBackend, state::NaSquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CPUBlob)
    pred  = inputs[1]
    label = inputs[2]

    data_type = eltype(pred)
    n = length(pred)
    num = get_num(pred)

    # fill diff with zeros
    erase!(diff)
    # diff.data <- weight * (pred.data - label.data) / num
    BLAS.axpy!(n, convert(data_type, state.layer.weight/num), pred.data, 1, diff.data, 1)
    BLAS.axpy!(n, convert(data_type, -state.layer.weight/num), label.data, 1, diff.data, 1)
    # handle the missing values
    rmna!(label.data, diff.data, state.NAvalue)
  end

  # the "label" also needs gradient
  if isa(diffs[2], CPUBlob)
    copy!(diffs[2], diff)
    BLAS.scal!(n, -one(data_type), diffs[2].data, 1)
  end
end

# removes value in y[i] if x[i] is missing x and y may be the same array
# TODO: write some fancy macro that handles both x[i] == NAvalue and x[i] == NaN
# not sure: isnan is vectorizable?
# TODO?: at some point we have to make sure that both inputs have the same size.
function rmna!{T,N}(x::Array{T,N}, y::Array{T,N}, NAvalue::T = -9999.0)
    if isnan(NAvalue)
        @simd for i in eachindex(x)
            @inbounds isnan(x[i])     && (y[i] = zero(T))
        end
    else
        @simd for i in eachindex(x)
            @inbounds x[i] == NAvalue && (y[i] = zero(T))
        end
    end
    nothing
end
