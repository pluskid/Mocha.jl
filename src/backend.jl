export Backend, CPUBackend, AbstractCuDNNBackend
export init, shutdown

abstract Backend
function init(backend::Backend)
end
function shutdown(backend::Backend)
end

type CPUBackend <: Backend; end

# This is forward declaration to allow some code to compile
# (especially testing codes) even if CUDA module is completely
# disabled. See test/layers/pooling.jl for example.
abstract AbstractCuDNNBackend <: Backend
