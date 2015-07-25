export GPUBackend

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
  softmax_loss_backward_float,
  softmax_loss_backward_double,
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
  elem_pow_fi,
  elem_pow_di,
  elem_pow_ff,
  elem_pow_dd,
  elem_log_double,
  elem_log_float,

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
)

function shutdown(mocha :: MochaKernels)
  CUDA.unload(mocha.mod)
end

type GPUBackend <: AbstractGPUBackend
  param_registry :: ParameterRegistry
  initialized    :: Bool
  cu_ctx         :: CUDA.CuContext
  cublas_ctx     :: CuBLAS.Handle
  cudnn_ctx      :: CuDNN.Handle

  mocha          :: MochaKernels

  GPUBackend() = new(ParameterRegistry(), false) # everything will be initialized later
end

function init(backend::GPUBackend)
  @assert backend.initialized == false

  @info("Initializing CuDNN backend...")
  CUDA.init()
  dev = CUDA.CuDevice(Config.cuda_dev_id)
  backend.cu_ctx = CUDA.create_context(dev)
  backend.cublas_ctx = CuBLAS.create()
  backend.cudnn_ctx = CuDNN.create()
  backend.mocha = MochaKernels()
  backend.initialized = true
  info("CuDNN backend initialized!")
end

function shutdown(backend::GPUBackend)
  @assert backend.initialized == true

  @info("Shutting down CuDNN backend...")
  # NOTE: destroy should be in reverse order of init
  shutdown(backend.mocha)
  CuDNN.destroy(backend.cudnn_ctx)
  CuBLAS.destroy(backend.cublas_ctx)
  CUDA.destroy(backend.cu_ctx)
  backend.initialized = false
  @info("CuDNN Backend shutdown finished!")
end
