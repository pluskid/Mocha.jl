for (functor, impl) in ((ElementWiseFunctors.Add, :(CuVec.add!)),
                        (ElementWiseFunctors.Subtract, :(CuVec.sub!)),
                        (ElementWiseFunctors.Multiply, :(CuVec.mul!)),
                        (ElementWiseFunctors.Divide, :(CuVec.div!)))
  @eval begin
    function forward(sys::System{CuDNNBackend}, state::ElementWiseLayerState{$functor},
        inputs::Vector{Blob})

      input1 = inputs[1]
      input2 = inputs[2]
      output = state.blobs[1]
      data_type = eltype(input1)
      len = length(input1)

      CuBLAS.copy(sys.backend.cublas_ctx, data_type, len, input1.ptr.p, 1, output.ptr.p, 1)
      $impl(sys, output, input2)
    end
  end
end

function backward(sys::System{CuDNNBackend}, state::ElementWiseLayerState{ElementWiseFunctors.Subtract},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  if !isa(diffs[1], NullBlob)
    copy!(diffs[1], state.blobs_diff[1])
  end
  if !isa(diffs[2], NullBlob)
    copy!(diffs[2], state.blobs_diff[1])
    CuBLAS.scal(sys.backend.cublas_ctx, length(diffs[2]), convert(eltype(diffs[2]),-1), 
        diffs[2].ptr, 1)
  end
end
function backward(sys::System{CuDNNBackend}, state::ElementWiseLayerState{ElementWiseFunctors.Multiply},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      copy!(diffs[i], state.blobs_diff[1])
      CuVec.mul!(sys, diffs[i], inputs[i%2 + 1])
    end
  end
end
function backward(sys::System{CuDNNBackend}, state::ElementWiseLayerState{ElementWiseFunctors.Divide},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  data_type = eltype(inputs[1])
  len = length(inputs[1])

  if !isa(diffs[1], NullBlob)
    copy!(diffs[1], state.blobs_diff[1])
    CuVec.div!(sys, diffs[1], inputs[2])
  end
  if !isa(diffs[2], NullBlob)
    copy!(diffs[2], state.blobs_diff[1])
    CuVec.mul!(sys, diffs[2], state.blobs[1])
    CuVec.div!(sys, diffs[2], inputs[2])
    CuBLAS.scal(sys.backend.cublas_ctx, len, convert(data_type,-1), diffs[2].ptr, 1)
  end
end
