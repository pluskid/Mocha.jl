############################################################
# Rectified-Linear
############################################################
function forward(sys :: System{CuDNNBackend}, neuron :: Neurons.ReLU, output :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = sys.backend.mocha.relu_forward_float
  elseif data_type == Float64
    kernel = sys.backend.mocha.relu_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(output.ptr.p, blob_dim...))
end

function backward(sys :: System{CuDNNBackend}, neuron :: Neurons.ReLU, output :: Blob, gradient :: Blob)
  cuda_dim, blob_dim = cuda_geometry(neuron, output)
  data_type = eltype(output)
  if data_type == Float32
    kernel = sys.backend.mocha.relu_backward_float
  elseif data_type == Float64
    kernel = sys.backend.mocha.relu_backward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, cuda_dim..., tuple(output.ptr.p, gradient.ptr.p, blob_dim...))
end


