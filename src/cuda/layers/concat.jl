function forward(backend::GPUBackend, state::ConcatLayerState, inputs::Vector{Blob})
  output = state.blobs[1]
  shifts = zeros(Int, 4)
  for i = 1:length(inputs)
    copy_to_shifted!(backend, output, inputs[i], shifts)
    shifts[state.layer.dim] += size(inputs[i], state.layer.dim)
  end
end

function backward(backend::GPUBackend, state::ConcatLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  grad = state.blobs_diff[1]
  shifts = zeros(Int, 4)
  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      copy_from_shifted!(backend, diffs[i], grad, shifts)
    end
    shifts[state.layer.dim] += size(inputs[i], state.layer.dim)
  end
end
