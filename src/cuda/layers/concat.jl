function forward(backend::GPUBackend, state::ConcatLayerState, inputs::Vector{Blob})
  output = state.blobs[1]
  shift = 0
  for i = 1:length(inputs)
    copy_to_shifted!(backend, output, inputs[i], (state.layer.dim, shift))
    shift += size(inputs[i], state.layer.dim)
  end
end

function backward(backend::GPUBackend, state::ConcatLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  grad = state.blobs_diff[1]
  shift = 0
  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      copy_from_shifted!(backend, diffs[i], grad, (state.layer.dim, shift))
    end
    shift += size(inputs[i], state.layer.dim)
  end
end
