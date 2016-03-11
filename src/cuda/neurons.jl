#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function cuda_geometry(:: ActivationFunction, output :: Blob)
  len = length(output)

  x_block = round(Int, ceil(convert(Float64, len)/1024));
  return ((x_block,1024), (len,))
end

############################################################
# Rectified-Linear
############################################################
function forward(backend :: GPUBackend, neuron :: Neurons.ReLU, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).relu_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).relu_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(convert(data_type, neuron.epsilon), get_ptr(output).p, blob_dim...), get_stream(backend))
end

function backward(backend :: GPUBackend, neuron :: Neurons.ReLU, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).relu_backward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).relu_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(convert(data_type, neuron.epsilon), get_ptr(output).p, get_ptr(gradient).p, blob_dim...),
                get_stream(backend))
end

############################################################
# Leaky Rectified-Linear
############################################################
function forward(backend :: GPUBackend, neuron :: Neurons.LReLU, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).lrelu_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).lrelu_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(get_ptr(output).p, blob_dim...), get_stream(backend))
end

function backward(backend :: GPUBackend, neuron :: Neurons.LReLU, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).lrelu_backward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).lrelu_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(get_ptr(output).p, get_ptr(gradient).p, blob_dim...), get_stream(backend))
end


function forward(backend :: GPUBackend, neuron :: Neurons.Sigmoid, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).sigmoid_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).sigmoid_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(get_ptr(output).p, blob_dim...), get_stream(backend))
end

function backward(backend :: GPUBackend, neuron :: Neurons.Sigmoid, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).sigmoid_backward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).sigmoid_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(get_ptr(output).p, get_ptr(gradient).p, blob_dim...), get_stream(backend))
end


function forward(backend :: GPUBackend, neuron :: Neurons.Tanh, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).tanh_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).tanh_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(get_ptr(output).p, blob_dim...), get_stream(backend))
end

function backward(backend :: GPUBackend, neuron :: Neurons.Tanh, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = get_mocha(backend).tanh_backward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).tanh_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(get_ptr(output).p, get_ptr(gradient).p, blob_dim...), get_stream(backend))
end


## Exponential

function forward(backend :: GPUBackend, neuron :: Neurons.Exponential, output :: Blob)
  CuVec.exp!(backend, output)
end

function backward(backend :: GPUBackend, neuron :: Neurons.Exponential, output :: Blob, gradient :: Blob)
  CuVec.mul!(backend, gradient, output)
end
