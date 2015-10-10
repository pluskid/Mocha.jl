using OpenCL

export ClTensorBlob

immutable ClTensorBlob{T<:AbstractFloat,N} <: Blob{T,N}
  # TODO
end
function ClTensorBlob{T<:AbstractFloat,N}(dtype::Type{T}, dims::NTuple{N,Int})
  # TODO
  Void()
end

length(b :: ClTensorBlob) = Void()
size(b :: ClTensorBlob) = Void()

function copy!{T}(dst :: ClTensorBlob{T}, src :: Array{T})
  # TODO
  Void
end
function copy!{T}(dst :: Array{T}, src :: ClTensorBlob{T})
  # TODO
  Void()
end
function copy!{T}(dst :: ClTensorBlob{T}, src :: ClTensorBlob{T})
  # TODO
  Void()
end
function fill!{T}(dst :: ClTensorBlob{T}, val)
  # TODO
  Void()
end
function erase!{T}(dst :: ClTensorBlob{T})
  # TODO
  Void()
end

function make_blob{N}(backend::OpenCLBackend, data_type::Type, dims::NTuple{N,Int})
  # TODO
  Void()
end
function reshape_blob{T,N}(backend::OpenCLBackend, blob::ClTensorBlob{T}, dims::NTuple{N,Int})
  # TODO
  Void()
end
function destroy(blob :: ClTensorBlob)
  # TODO
  Void()
end

