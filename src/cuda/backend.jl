export GPUBackend, MultiGPUType
export get_cublas_ctx, get_cudnn_ctx, get_stream, get_mocha

macro defkernels(kernels...)
  field_defs = map(kernels) do ker
    :($ker :: CUDA.CuFunction)
  end
  type_body = Expr(:block, field_defs...)

  field_inits = map(kernels) do ker
    :(kernels.$ker = CUDA.CuFunction(mod, $(string(ker))))
  end
  field_init_block = Expr(:block, field_inits...)

  quote
    type $(esc(:MochaKernels))
      mod :: CUDA.CuModule

      $type_body

      $(esc(:MochaKernels))() = begin
        mod_dir = joinpath(dirname(@__FILE__), "kernels")
        mod_path = joinpath(mod_dir, "kernels.ptx")

        # check that our module is up-to-date
        if !isfile(mod_path)
          error("Mocha CUDA kernels not found, see the documents of BACKEND on how to compile the kernels")
        else
          mod_mtime = stat(mod_path).mtime
          impl_files = glob(mod_dir, r".*.impl$")
          for i = 1:length(impl_files)
            if stat(joinpath(mod_dir, impl_files[i])).mtime > mod_mtime
              error("Mocha CUDA kernels not up-to-date. Please re-compile (see documents of BACKEND)")
            end
          end
        end

        mod = CUDA.CuModule(mod_path)
        kernels = new(mod)

        $field_init_block

        return kernels
      end
    end
  end
end

@defkernels(
  logistic_loss_forward_float,
  logistic_loss_forward_double,
  binary_cross_entropy_loss_forward_float,
  binary_cross_entropy_loss_forward_double,
  binary_cross_entropy_loss_backward_float,
  binary_cross_entropy_loss_backward_double,
  softmax_loss_backward_float,
  softmax_loss_backward_double,
  hinge_loss_forward_float,
  hinge_loss_forward_double,
  hinge_loss_backward_float,
  hinge_loss_backward_double,
  relu_forward_float,
  relu_forward_double,
  relu_backward_float,
  relu_backward_double,
  lrelu_forward_float,
  lrelu_forward_double,
  lrelu_backward_float,
  lrelu_backward_double,
  sigmoid_forward_float,
  sigmoid_forward_double,
  sigmoid_backward_float,
  sigmoid_backward_double,
  tanh_forward_float,
  tanh_forward_double,
  tanh_backward_float,
  tanh_backward_double,
  accuracy_forward_float,
  accuracy_forward_double,
  binary_accuracy_forward_float,
  binary_accuracy_forward_double,
  argmax_forward_float,
  argmax_forward_double,
  index2onehot_forward_float,
  index2onehot_forward_double,

  add_scal_float,
  add_scal_double,
  mul_scal_float,
  mul_scal_double,
  elem_add_float,
  elem_add_double,
  elem_mul_float,
  elem_mul_double,
  elem_sub_float,
  elem_sub_double,
  elem_div_float,
  elem_div_double,
  elem_div2_float,
  elem_div2_double,
  elem_mean_float,
  elem_mean_double,
  elem_pow_fi,
  elem_pow_di,
  elem_pow_ff,
  elem_pow_dd,
  elem_log_double,
  elem_log_float,
  elem_exp_double,
  elem_exp_float,

  max_channel_pooling_forward_float,
  max_channel_pooling_forward_double,
  max_channel_pooling_backward_float,
  max_channel_pooling_backward_double,

  dense_to_padded_float,
  dense_to_padded_double,
  padded_to_dense_float,
  padded_to_dense_double,

  copy_to_shifted_float,
  copy_to_shifted_double,
  copy_from_shifted_float,
  copy_from_shifted_double,

  dropout_init,
  dropout_alloc_size,
  dropout_forward_float,
  dropout_forward_double,
  dropout_backward_float,
  dropout_backward_double,

  l1_forward_float,
  l1_forward_double,
  l1_backward_float,
  l1_backward_double,

  stdnormal_init,
  stdnormal_alloc_size,
  stdnormal_forward_float,
  stdnormal_forward_double,
)

function shutdown(mocha :: MochaKernels)
  CUDA.unload(mocha.mod)
end

type GPUBackend <: AbstractGPUBackend
  param_registry :: ParameterRegistry
  initialized    :: Bool
  cur_dev        :: CudaRT.CudaDevice
  dev_count      :: Int
  streams        :: Array{CudaRT.CudaStream}
  cublas_ctxs    :: Array{CuBLAS.Handle}
  cudnn_ctxs     :: Array{CuDNN.Handle}
  mochas         :: Array{MochaKernels}

  GPUBackend() = new(ParameterRegistry(), false) # everything will be initialized later
end

function set_dev_id(device::CudaRT.CudaDevice, id::Int)
  device.ordinal = id
  CudaRT.set_device(device)
end

function set_dev_id(backend::GPUBackend, id::Int)
  set_dev_id(backend.cur_dev, id)
end

function set_dev(backend::GPUBackend, id::Int)
  set_dev_id(backend, id)
  @inbounds CuBLAS.set_stream(backend.cublas_ctxs[id + 1], backend.streams[id + 1])
  @inbounds CuDNN.set_stream(backend.cublas_ctxs[id + 1], backend.streams[id + 1])
end

function get_cublas_ctx(backend::GPUBackend)
  @inbounds return backend.cublas_ctxs[backend.cur_dev.ordinal + 1]
end

function get_cudnn_ctx(backend::GPUBackend)
  @inbounds return backend.cudnn_ctxs[backend.cur_dev.ordinal + 1]
end

function get_stream(backend::GPUBackend)
  @inbounds return backend.streams[backend.cur_dev.ordinal + 1]
end

function get_mocha(backend::GPUBackend)
  @inbounds return backend.mochas[backend.cur_dev.ordinal + 1]
end

# sync all GPU's streams
function sync(backend::GPUBackend)
  for i=1:backend.dev_count
    CudaRT.sync_stream(backend.streams[i])
  end
end

function init(backend::GPUBackend)
  @assert backend.initialized == false

  @info("Initializing CuDNN backend...")
  @assert Config.cuda_dev_id < Config.cuda_dev_count
  backend.cur_dev = CudaRT.CudaDevice(Config.cuda_dev_id)
  backend.dev_count = Config.cuda_dev_count
  backend.streams = Array(CudaRT.CudaStream, backend.dev_count)
  backend.cublas_ctxs = Array(CuBLAS.Handle, backend.dev_count)
  backend.cudnn_ctxs = Array(CuDNN.Handle, backend.dev_count)
  backend.mochas = Array(MochaKernels, backend.dev_count)
  @inbounds for i=1:backend.dev_count
    CudaRT.set_device(CudaDevice(i - 1))
    backend.streams[i] = CudaRT.create_stream()
    backend.cublas_ctxs[i] = CuBLAS.create()
    backend.cudnn_ctxs[i] = CuDNN.create()
    backend.mochas[i] = MochaKernels()
  end
  set_dev(backend, Config.cuda_dev_id)
  backend.initialized = true
  info("CuDNN backend initialized!")
end

function shutdown(backend::GPUBackend)
  @assert backend.initialized == true

  @info("Shutting down CuDNN backend...")
  # NOTE: destroy should be in reverse order of init
  map(shutdown, backend.mochas)
  map(CuDNN.destroy, backend.cudnn_ctxs)
  map(CuBLAS.destroy, backend.cublas_ctxs)
  map(CudaRT.destroy, backend.streams)
  backend.initialized = false
  @info("CuDNN Backend shutdown finished!")
end

type MultiGPUType{T}
  elems   :: Array{T}
  cur_dev :: CudaRT.CudaDevice
end
function MultiGPUType{T}(backend::GPUBackend, dtype::Type{T})
  elems = Array(T, backend.dev_count)
  return MultiGPUType(elems, backend.cur_dev)
end
get_elem(multi :: MultiGPUType) = @inbounds return multi.elems[multi.cur_dev.ordinal + 1]
ndev(multi :: MultiGPUType) = return length(multi.elems)
