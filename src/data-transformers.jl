export DataTransformerType, DataTransformerState, DataTransformers
export setup, forward, shutdown

@compat abstract type DataTransformerType end
@compat abstract type DataTransformerState end

module DataTransformers
using ..Mocha
using Compat

immutable SubMean <: DataTransformerType
  mean_file :: AbstractString
  mean_blob :: Blob
end
SubMean(;mean_file="", mean_blob=NullBlob()) = SubMean(mean_file, mean_blob)

immutable Scale <: DataTransformerType
  scale :: AbstractFloat
end
Scale(;scale=1.0) = Scale(scale)

end # module DataTransformers

################################################################################
# Subtract Mean
################################################################################
type SubMeanState <: DataTransformerState
  transformer :: DataTransformers.SubMean
  mean_blob   :: Blob
  multiplier  :: Blob
end
function setup(backend::Backend, transformer::DataTransformers.SubMean, input::Blob)
  if isa(transformer.mean_blob, NullBlob)
    h5open(transformer.mean_file, "r") do h5
      mean_data = read(h5, "mean")
      mean_dims = size(mean_data)
      if length(mean_dims) == ndims(input) - 1
        mean_dims = tuple(mean_dims..., 1)
      end
      mean_blob = make_blob(backend, eltype(input), mean_dims)
      copy!(mean_blob, convert(Array{eltype(input)}, mean_data))
    end
  else
    mean_blob = make_blob(backend, eltype(input), size(transformer.mean_blob))
    copy!(mean_blob, transformer.mean_blob)
  end

  @assert size(mean_blob)[1:end-1] == size(input)[1:end-1]

  multiplier = make_blob(backend, eltype(input), get_num(input))
  fill!(multiplier, 1)

  return SubMeanState(transformer, mean_blob, multiplier)
end
function forward(backend::CPUBackend, state::SubMeanState, input::Blob)
  fea_dim = get_fea_size(input)
  num     = get_num(input)
  RawBLAS.gemm!('N', 'N', fea_dim, num, 1, convert(eltype(input), -1),
      state.mean_blob.data, fea_dim, state.multiplier.data, 1, convert(eltype(input), 1),
      input.data, fea_dim)
end
function shutdown(backend::Backend, state::SubMeanState)
  destroy(state.mean_blob)
  destroy(state.multiplier)
end

################################################################################
# Scale
################################################################################
type ScaleState{T} <: DataTransformerState
  transformer :: DataTransformers.Scale
  scale       :: T
end
function setup(backend::Backend, transformer::DataTransformers.Scale, input::Blob)
  return ScaleState(transformer, convert(eltype(input), transformer.scale))
end
function forward(backend::CPUBackend, state::ScaleState, input::Blob)
  BLAS.scal!(length(input.data), state.scale, input.data, 1)
end
function shutdown(backend::Backend, state::ScaleState)
end
