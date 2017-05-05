############################################################
# Split Layer
############################################################
@defstruct SplitLayer Layer (
  name :: AbstractString = "split",
  no_copy :: Bool = false,
  (tops :: Vector{Symbol} = Symbol[], length(tops) >= 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 1),
)
@characterize_layer(SplitLayer,
  can_do_bp => true
)

type SplitLayerState{N} <: LayerState
  layer      :: SplitLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::SplitLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  N = length(layer.tops)

  if layer.no_copy
    # directly re-use the input blob
    blobs = Blob[inputs[1] for i = 1:N]
  else
    blobs = Array{Blob}(N)
    blobs[1] = inputs[1]
    for i = 2:N
      blobs[i] = make_blob(backend, eltype(inputs[1]), size(inputs[1]))
    end
  end
  blobs_diff = Array{Blob}(N)
  blobs_diff[1] = diffs[1] # re-use the first backward blob
  for i = 2:N
    if isa(diffs[1], NullBlob)
      blobs_diff[i] = NullBlob()
    else
      blobs_diff[i] = make_blob(backend, eltype(inputs[1]), size(inputs[1]))
    end
  end

  return SplitLayerState{N}(layer, blobs, blobs_diff)
end
function shutdown(backend::Backend, state::SplitLayerState)
  # some blobs are shared, but never mind, blob destroy function has
  # a guard that does not cause problems on double destroying
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
end

function forward(backend::Backend, state::SplitLayerState, inputs::Vector{Blob})
  if !state.layer.no_copy
    for i = 2:length(state.blobs)
      copy!(state.blobs[i], inputs[1])
    end
  end
end

function backward{N}(backend::CPUBackend, state::SplitLayerState{N}, inputs::Vector{Blob}, diffs::Vector{Blob})
  if !isa(diffs[1], NullBlob)
    diff = diffs[1]
    len = length(diff)
    one = convert(eltype(diff), 1)
    for i = 2:N
      BLAS.axpy!(len, one, state.blobs_diff[i].data, 1, diff.data, 1)
    end
  end
end

