export Backend, CPU, GPU
export Blob, NullBlob, CPUBlob

abstract Backend

type CPU <: Backend; end
type CUDNN <: Backend; end

abstract Blob

# used to indicate an empty blob that holds nothing
type NullBlob <: Blob; end

# blob that holds data in CPU memory
type CPUBlob <: Blob
    name :: String
    data :: Array
end

