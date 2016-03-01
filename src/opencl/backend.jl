export OpenCLBackend
# TODO: define kernels
# TODO: define shutdown function for kernels

type OpenCLBackend <: AbstractGPUBackend
  param_registry :: ParameterRegistry

  platform       :: cl.Platform
  device         :: cl.Device
  initialized    :: Bool

  context        :: cl.Context
  queue          :: cl.CmdQueue

  function OpenCLBackend(platform_id = Config.opencl_platform_id,
                         device_id   = Config.opencl_dev_id)
    platform    = cl.platforms()[platform_id+1]
    device      = cl.devices(platform)[device_id+1]
    new(ParameterRegistry(), platform, device, false) # everything will be initialized later
  end
end

function init(backend :: OpenCLBackend)
  @assert backend.initialized == false

  @info("Initializing OpenCL backend...")
  backend.context     = cl.Context(backend.device)
  backend.queue       = cl.CmdQueue(backend.context)

  clblas.setup()

  backend.initialized = true
  @info("OpenCL backend initialized!")
end

function shutdown(backend :: OpenCLBackend)
  @assert backend.initialized = true

  @info("Shutting down OpenCL backend...")
  # NOTE: destroy should be in reverse order of init
  backend.initialized = false

  clblas.teardown()

  registry_reset(backend)
  finalize(backend.queue)
  finalize(backend.context)
  @info("OpenCL backend shutdown finished!")
end
