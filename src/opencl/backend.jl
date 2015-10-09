export OpenCLBackend

# TODO: define kernels
# TODO: define shutdown function for kernels

type OpenCLBackend <: AbstractGPUBackend
  initialized    :: Bool
  # TODO

  OpenCLBackend() = new(false) # everything will be initialized later
end

function init(backend :: OpenCLBackend)
  @assert backend.initialized == false

  @info("STUB: Initializing OpenCL backend...")
  # TODO

  backend.initialized = true
  @info("STUB: OpenCL backend initialized!")
end

function shutdown(backend :: OpenCLBackend)
  @assert backend.initialized = true

  @info("STUB: Shutting down OpenCL backend...")
  # TODO

  backend.initialized = false
  @info("STUB: OpenCL backend shutdown finished!")
end
