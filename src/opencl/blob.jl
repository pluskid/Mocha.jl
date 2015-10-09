using OpenCL

export ClTensorBlob

immutable ClTensorBlob{T<:AbstractFloat,N} <: Blob{T,N}
  # TODO
end
function ClTensorBlob{T<:AbstractFloat,N}(dtype::Type{T}, dims::NTuple{N,Int})
  # TODO
  nothing
end

length(b :: ClTensorBlob) = nothing()
size(b :: ClTensorBlob) = nothing()

function copy!{T}(dst :: ClTensorBlob{T}, src :: Array{T})
  # TODO
  nothing
end
function copy!{T}(dst :: Array{T}, src :: ClTensorBlob{T})
  # TODO
  nothing
end
function copy!{T}(dst :: ClTensorBlob{T}, src :: ClTensorBlob{T})
  # TODO
  nothing
end
function fill!{T}(dst :: ClTensorBlob{T}, val)
  # TODO
  nothing
end
function erase!{T}(dst :: ClTensorBlob{T})
  # TODO
  nothing
end

function make_blob{N}(backend::OpenCLBackend, data_type::Type, dims::NTuple{N,Int})
  # TODO
  nothing
end
function reshape_blob{T,N}(backend::OpenCLBackend, blob::ClTensorBlob{T}, dims::NTuple{N,Int})
  # TODO
  nothing
end
function destroy(blob :: ClTensorBlob)
  # TODO
  nothing
end

