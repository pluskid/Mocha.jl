# a simplified CudaRT module, adapted from github.com/JuliaGPU/CUDArt.jl
export CudaRT
module CudaRT
export CudaStream
export set_device_count, get_device_count, set_device, get_device, create_stream, sync_stream, cuda_null_stream

@windows? (
begin
    const libcudart = Libdl.find_library(["cudart64_75", "cudart64_70", 
        "cudart64_65", "cudart32_75", "cudart32_70", "cudart32_65"], [""])
end
: # linux or mac
begin
    const libcudart = Libdl.find_library(["libcudart"], [])
end)

using Compat
const cuda_error_descriptions = @compat(Dict{Int,AbstractString}(
        0 => "cudaSuccess",
        1 => "cudaErrorMissingConfiguration",
        2 => "cudaErrorMemoryAllocation",
        3 => "cudaErrorInitializationError",
        4 => "cudaErrorLaunchFailure",
        5 => "cudaErrorPriorLaunchFailure",
        6 => "cudaErrorLaunchTimeout",
        7 => "cudaErrorLaunchOutOfResources",
        8 => "cudaErrorInvalidDeviceFunction",
        9 => "cudaErrorInvalidConfiguration",
        10 => "cudaErrorInvalidDevice",
        11 => "cudaErrorInvalidValue",
        12 => "cudaErrorInvalidPitchValue",
        13 => "cudaErrorInvalidSymbol",
        14 => "cudaErrorMapBufferObjectFailed",
        15 => "cudaErrorUnmapBufferObjectFailed",
        16 => "cudaErrorInvalidHostPointer",
        17 => "cudaErrorInvalidDevicePointer",
        18 => "cudaErrorInvalidTexture",
        19 => "cudaErrorInvalidTextureBinding",
        20 => "cudaErrorInvalidChannelDescriptor",
        21 => "cudaErrorInvalidMemcpyDirection",
        22 => "cudaErrorAddressOfConstant",
        23 => "cudaErrorTextureFetchFailed",
        24 => "cudaErrorTextureNotBound",
        25 => "cudaErrorSynchronizationError",
        26 => "cudaErrorInvalidFilterSetting",
        27 => "cudaErrorInvalidNormSetting",
        28 => "cudaErrorMixedDeviceExecution",
        29 => "cudaErrorCudartUnloading",
        30 => "cudaErrorUnknown",
        31 => "cudaErrorNotYetImplemented",
        32 => "cudaErrorMemoryValueTooLarge",
        33 => "cudaErrorInvalidResourceHandle",
        34 => "cudaErrorNotReady",
        35 => "cudaErrorInsufficientDriver",
        36 => "cudaErrorSetOnActiveProcess",
        37 => "cudaErrorInvalidSurface",
        38 => "cudaErrorNoDevice",
        39 => "cudaErrorECCUncorrectable",
        40 => "cudaErrorSharedObjectSymbolNotFound",
        41 => "cudaErrorSharedObjectInitFailed",
        42 => "cudaErrorUnsupportedLimit",
        43 => "cudaErrorDuplicateVariableName",
        44 => "cudaErrorDuplicateTextureName",
        45 => "cudaErrorDuplicateSurfaceName",
        46 => "cudaErrorDevicesUnavailable",
        47 => "cudaErrorInvalidKernelImage",
        48 => "cudaErrorNoKernelImageForDevice",
        49 => "cudaErrorIncompatibleDriverContext",
        50 => "cudaErrorPeerAccessAlreadyEnabled",
        51 => "cudaErrorPeerAccessNotEnabled",
        54 => "cudaErrorDeviceAlreadyInUse",
        55 => "cudaErrorProfilerDisabled",
        56 => "cudaErrorProfilerNotInitialized",
        57 => "cudaErrorProfilerAlreadyStarted",
        58 => "cudaErrorProfilerAlreadyStopped",
        59 => "cudaErrorAssert",
        60 => "cudaErrorTooManyPeers",
        61 => "cudaErrorHostMemoryAlreadyRegistered",
        62 => "cudaErrorHostMemoryNotRegistered",
        63 => "cudaErrorOperatingSystem",
        64 => "cudaErrorPeerAccessUnsupported",
        65 => "cudaErrorLaunchMaxDepthExceeded",
        66 => "cudaErrorLaunchFileScopedTex",
        67 => "cudaErrorLaunchFileScopedSurf",
        68 => "cudaErrorSyncDepthExceeded",
        69 => "cudaErrorLaunchPendingCountExceeded",
        70 => "cudaErrorNotPermitted",
        71 => "cudaErrorNotSupported",
        72 => "cudaErrorHardwareStackError",
        73 => "cudaErrorIllegalInstruction",
        74 => "cudaErrorMisalignedAddress",
        75 => "cudaErrorInvalidAddressSpace",
        76 => "cudaErrorInvalidPc",
        77 => "cudaErrorIllegalAddress",
        78 => "cudaErrorInvalidPtx",
        79 => "cudaErrorInvalidGraphicsContext",
        0x7f => "cudaErrorStartupFailure",
        10000 => "cudaErrorApiFailureBase",
))

immutable CudaError <: Exception
  code :: Int
end
import Base.show
show(io::IO, error::CudaError) = print(io, cuda_error_descriptions[error.code])

macro cudacall(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), $libcudart), Cint, $argtypes, $(args...)  )
    if round(Int, _curet) != 0
      throw(CudaError(round(Int, _curet)))
    end
  end
end

############################################################
# CUDART devices
############################################################
dev_count = 0
function set_dev_count(count::Int)
  global dev_count = count
end
function get_dev_count()
  return dev_count
end

current_dev = 0
function set_device(device_id::Int)
  @cudacall(:cudaSetDevice, (Cint,), device_id)
  global current_dev = device_id
end
function get_device()
  return current_dev
end


############################################################
# CUDART streams
############################################################
typealias CudaStream Ptr{Void}

function create_stream()
  a = CudaStream[0]
  @cudacall(:cudaStreamCreate, (Ptr{CudaStream},), a)
  return a[1]
end

function destroy_stream(stream :: CudaStream)
  if (stream != cuda_null_stream())
    @cudacall(:cudaStreamDestroy, (CudaStream,), stream)
  end
end

function sync_stream(stream :: CudaStream)
  @cudacall(:cudaStreamSynchronize, (CudaStream,), stream)
end

cuda_null_stream() = C_NULL

destroy(stream :: CudaStream) = destroy_stream(stream)

end # module CudaRT