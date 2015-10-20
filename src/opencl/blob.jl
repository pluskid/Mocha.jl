export ClTensorBlob

immutable ClTensorBlob{T<:AbstractFloat,N} <: Blob{T,N}
  backend :: OpenCLBackend
  buffer  :: cl.Buffer
  shape   :: NTuple{N, Int}
end
function ClTensorBlob{T<:AbstractFloat, N}(backend::OpenCLBackend, dtype::Type{T}, dims::NTuple{N,Int})
  len = prod(dims)
  buffer = cl.Buffer(T, backend.context, len)
  ClTensorBlob{T,N}(backend, buffer, dims)
end

length(b :: ClTensorBlob) = length(b.buffer)
size(b :: ClTensorBlob) = b.shape
queue(b :: ClTensorBlob) = b.backend.queue

function copy!{T}(dst :: ClTensorBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  cl.copy!(queue(dst), dst.buffer, src)
end
function copy!{T}(dst :: Array{T}, src :: ClTensorBlob{T})
  @assert length(dst) == length(src)
  cl.copy!(queue(src), dst, src.buffer)
end
function copy!{T}(dst :: ClTensorBlob{T}, src :: ClTensorBlob{T})
  @assert dst.backend === src.backend
  @assert length(dst) == length(src)
  cl.copy!(queue(dst), dst.buffer, src.buffer)
end
function fill!{T}(dst :: ClTensorBlob{T}, val)
  cl.enqueue_fill_buffer(queue(dst), dst.buffer, val, UInt(0),
                         UInt(sizeof(dst.buffer)), Void())
end
function erase!{T}(dst :: ClTensorBlob{T})
  fill!(dst, zero(T))
end

function make_blob{N}(backend::OpenCLBackend, data_type::Type, dims::NTuple{N,Int})
  ClTensorBlob(backend, data_type, dims)
end
function reshape_blob{T,N}(backend::OpenCLBackend, blob::ClTensorBlob{T}, dims::NTuple{N,Int})
  @assert backend === blob.backend
  @assert prod(dims) == length(blob)
  ClTensorBlob{T,N}(backend, blob.buffer, dims)
end
function destroy(blob :: ClTensorBlob)
  if blob.buffer.valid
    cl.release!(blob.buffer)
  end
end

