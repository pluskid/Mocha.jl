export DataTransformerType, DataTransformerState, DataTransformers
export setup, forward, shutdown

abstract DataTransformerType
abstract DataTransformerState

module DataTransformers
using ..Mocha

immutable SubMean <: DataTransformerType
  mean_file :: String
  mean_blob :: Blob
end
SubMean(;mean_file="", mean_blob=NullBlob()) = SubMean(mean_file, mean_blob)

immutable Scale <: DataTransformerType
  scale :: FloatingPoint
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
      if length(mean_dims) == 3
        mean_dims = [mean_dims..., 1]
      end
      mean_blob = make_blob(backend, eltype(input), mean_dims...)
      copy!(mean_blob, convert(Array{eltype(input)}, mean_data))
    end
  else
    mean_blob = make_blob(backend, eltype(input), size(transformer.mean_blob))
    copy!(mean_blob, transformer.mean_blob)
  end

  @assert get_num(mean_blob) == 1
  @assert get_width(mean_blob) == get_width(input)
  @assert get_height(mean_blob) == get_height(input)
  @assert get_chann(mean_blob) == get_chann(input)

  multiplier = make_blob(backend, eltype(input), get_num(input), 1, 1, 1)
  fill!(multiplier, 1)

  return SubMeanState(transformer, mean_blob, multiplier)
end
function forward(backend::CPUBackend, state::SubMeanState, input::Blob)
  width, height, channels, num = size(input)
  fea_dim = width*height*channels
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
