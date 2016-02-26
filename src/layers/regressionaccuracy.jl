@defstruct RegressionAccuracyLayer Layer (
  name :: AbstractString = "regressionaccuracy",
  report_error :: Bool = false,
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(RegressionAccuracyLayer,
  is_sink    => true,
  has_stats  => true,
)

type RegressionAccuracyLayerState <: LayerState
  layer :: RegressionAccuracyLayer

  op_dim   :: Int
  accuracy :: Float64
  n_accum  :: Int
  etc      :: Any
end

function setup_etc(backend::CPUBackend, layer::RegressionAccuracyLayer, op_dim::Int, inputs)
  nothing
end

function setup(backend::Backend, layer::RegressionAccuracyLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  total_dim = ndims(inputs[1])
  dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
  @assert 1 <= dim <= total_dim
  @assert dim != total_dim

  etc = setup_etc(backend, layer, dim, inputs)
  return RegressionAccuracyLayerState(layer, dim, 0.0, 0, etc)
end
function shutdown(backend::CPUBackend, state::AccuracyLayerState)
end

function reset_statistics(state::RegressionAccuracyLayerState)
  state.n_accum = 0
  state.accuracy = 0.0
end

function dump_statistics(storage, state::RegressionAccuracyLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-accuracy", state.accuracy)
  if state.layer.report_error
    update_statistics(storage, "$(state.layer.name)-error", 1-state.accuracy)
  end
  if show
    accuracy = @sprintf("%.4f", state.accuracy)
    @info("  Test sample avg. (over all outputs  and sample) RMSE: $accuracy")
  end
end

function forward(backend::CPUBackend, state::RegressionAccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data
  dim_pre, dim_prob, dim_post = split_dims(pred, state.op_dim)
  accuracy = zeros(size(label,1),1)
  #for i = 1:dim_pre
    for j = 1:dim_post
        accuracy += (label[:,j] - pred[:,j]).^2.0
      end
    #  end
  @info("")  
  @info("---------------------------------------------------------")
  @info("Test sample RMSEs, all outputs:  ")
  @info("$(round(sqrt(accuracy'/dim_post),4))")
  @info("---------------------------------------------------------")
  state.accuracy = mean(sqrt(accuracy/dim_post))
  state.n_accum += dim_post
end

function backward(backend::Backend, state::RegressionAccuracyLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

