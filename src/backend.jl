export Backend, CPUBackend

abstract Backend
function init(backend::Backend)
end
function shutdown(backend::Backend)
end

type CPUBackend <: Backend; end

