function cuda_geometry(:: ActivationFunction, output :: Blob)
  width, height, channels, num = get_whcn(output)
  spatial_dim = width*height

  x_block = int(ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X));
  y_block = int(ceil(float64(channels)/CUDA.THREADS_PER_BLOCK_Y));
  z_block = int(ceil(float64(spatial_dim)/CUDA.THREADS_PER_BLOCK_Z));
  return (((x_block,y_block,z_block),(CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y,CUDA.THREADS_PER_BLOCK_Z)),
          (num, channels, spatial_dim))
end

############################################################
# Rectified-Linear
############################################################
function forward(backend :: GPUBackend, neuron :: Neurons.ReLU, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = backend.mocha.relu_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.relu_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(output.ptr.p, blob_dim...))
end

function backward(backend :: GPUBackend, neuron :: Neurons.ReLU, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = backend.mocha.relu_backward_float
  elseif data_type == Float64
    kernel = backend.mocha.relu_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(output.ptr.p, gradient.ptr.p, blob_dim...))
end


function forward(backend :: GPUBackend, neuron :: Neurons.Sigmoid, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = backend.mocha.sigmoid_forward_float
  elseif data_type == Float64
    kernel = backend.mocha.sigmoid_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(output.ptr.p, blob_dim...))
end

function backward(backend :: GPUBackend, neuron :: Neurons.Sigmoid, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = backend.mocha.sigmoid_backward_float
  elseif data_type == Float64
    kernel = backend.mocha.sigmoid_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(output.ptr.p, gradient.ptr.p, blob_dim...))
end

