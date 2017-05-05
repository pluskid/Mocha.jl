export ElementWiseFunctorType, ElementWiseFunctors
export get_num_args
@compat abstract type ElementWiseFunctorType{NArg} end

get_num_args{NArg}(::ElementWiseFunctorType{NArg}) = NArg

module ElementWiseFunctors
using ..Mocha
type Add <: ElementWiseFunctorType{2}
end
type Subtract <: ElementWiseFunctorType{2}
end
type Multiply <: ElementWiseFunctorType{2}
end
type Divide <: ElementWiseFunctorType{2}
end
end # module ElementWiseFunctors

############################################################
# Element-wise operation layer
############################################################
@defstruct ElementWiseLayer Layer (
  name :: AbstractString = "element-wise",
  operation :: ElementWiseFunctorType = ElementWiseFunctors.Add(),
  (tops :: Vector{Symbol} = Symbol[], length(tops) == 1),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == get_num_args(operation)),
)
@characterize_layer(ElementWiseLayer,
  can_do_bp => true
)

type ElementWiseLayerState{Op<:ElementWiseFunctorType} <: LayerState
  layer      :: ElementWiseLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::ElementWiseLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert all(map(i -> size(inputs[i]) == size(inputs[1]), 2:length(inputs)))
  blobs = Blob[make_blob(backend, eltype(inputs[1]), size(inputs[1]))]
  if all(map(b -> isa(b, NullBlob), diffs))
    blobs_diff = Blob[NullBlob()]
  else
    blobs_diff = Blob[make_blob(backend, eltype(inputs[1]), size(inputs[1]))]
  end

  return ElementWiseLayerState{typeof(layer.operation)}(layer, blobs, blobs_diff)
end
function shutdown(backend::Backend, state::ElementWiseLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
end

for (functor, op) in ((ElementWiseFunctors.Add, (+)),
                      (ElementWiseFunctors.Subtract, (-)),
                      (ElementWiseFunctors.Multiply, (*)),
                      (ElementWiseFunctors.Divide, (/)))
  @eval begin
    # I'm getting the following warning unless I extract the for loop
    # as a separate function with clear type annotations.
    # Warning: could not attach metadata for @simd loop.
    function functor_impl{T}(::$functor, input1::Array{T}, input2::Array{T}, output::Array{T})
      len = length(input1)
      @simd for i = 1:len
        @inbounds output[i] = $op(input1[i], input2[i])
      end
    end
    function forward(backend::CPUBackend, state::ElementWiseLayerState{$functor},
        inputs::Vector{Blob})

      input1 = inputs[1].data
      input2 = inputs[2].data
      output = state.blobs[1].data
      functor_impl(state.layer.operation, input1, input2, output)
    end
  end
end

function backward(backend::Backend, state::ElementWiseLayerState{ElementWiseFunctors.Add},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      copy!(diffs[i], state.blobs_diff[1])
    end
  end
end
function backward(backend::CPUBackend, state::ElementWiseLayerState{ElementWiseFunctors.Subtract},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  if !isa(diffs[1], NullBlob)
    copy!(diffs[1], state.blobs_diff[1])
  end
  if !isa(diffs[2], NullBlob)
    copy!(diffs[2], state.blobs_diff[1])
    BLAS.scal!(length(diffs[2]), convert(eltype(diffs[2]),-1), diffs[2].data, 1)
  end
end
function backward(backend::CPUBackend, state::ElementWiseLayerState{ElementWiseFunctors.Multiply},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  for i = 1:length(diffs)
    if !isa(diffs[i], NullBlob)
      copy!(diffs[i], state.blobs_diff[1])
      Vec.mul!(diffs[i].data, inputs[i%2 + 1].data)
    end
  end
end
function backward(backend::CPUBackend, state::ElementWiseLayerState{ElementWiseFunctors.Divide},
    inputs::Vector{Blob}, diffs::Vector{Blob})

  if !isa(diffs[1], NullBlob)
    copy!(diffs[1], state.blobs_diff[1])
    Vec.div!(diffs[1].data, inputs[2].data)
  end
  if !isa(diffs[2], NullBlob)
    copy!(diffs[2], state.blobs_diff[1])
    Vec.mul!(diffs[2].data, state.blobs[1].data)
    Vec.div!(diffs[2].data, inputs[2].data)
    BLAS.scal!(length(diffs[2]), convert(eltype(diffs[2]),-1), diffs[2].data, 1)
  end
end

