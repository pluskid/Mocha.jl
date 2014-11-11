export ActivationFunction
export Neurons

export forward, backward

############################################################
# Each activation function / neuron should be a sub type
# of ActivationFunction and should implement the following
# methods.
#
# - forward(sys :: System, neuron :: MyActivationFunction, output :: Blob)
#   Update the output blob by the activation function. Computation should be done in-place.
#   Note we call is "output" because an activation function is applied to the "output" of
#   a layer.
#
# - backward(sys :: System, neuron :: MyActivationFunction, output :: Blob, gradient :: Blob)
#   The gradient blob contains the gradient with respect to the output of the activation
#   function, update (IN-PLACE) the gradient to be with respect to the output of the activation
#   function.
############################################################

abstract ActivationFunction

############################################################
# A module to hold built-in activation functions to avoid
# messy namespace
############################################################
module Neurons
using ..Mocha.ActivationFunction
# Identity
type Identity <: ActivationFunction
end

# Rectified-Linear: ReLU(x) = max(x,0)
type ReLU <: ActivationFunction
end

# Sigmoid: Sigmoid(x) = 1 / (1 + exp(-x))
type Sigmoid <: ActivationFunction
end
end # module Neurons

############################################################
# Identity
############################################################
function forward(sys :: System, neuron :: Neurons.Identity, output :: Blob)
  # do nothing
end
function backward(sys :: System, neuron :: Neurons.Identity, output :: Blob, gradient :: Blob)
  # do nothing
end

function cuda_geometry(:: ActivationFunction, output :: Blob)
  width, height, channels, num = size(output)
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
function forward(sys :: System{CPUBackend}, neuron :: Neurons.ReLU, output :: Blob)
  output.data = max(output.data, 0)
end
function backward(sys :: System{CPUBackend}, neuron :: Neurons.ReLU, output :: Blob, gradient :: Blob)
  gradient.data .*= (output.data .> 0)
end

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

############################################################
# Sigmoid
############################################################
function forward(sys :: System{CPUBackend}, neuron :: Neurons.Sigmoid, output :: Blob)
  output.data = 1 ./ (1 + exp(-output.data))
end
function backward(sys :: System{CPUBackend}, neuron :: Neurons.Sigmoid, output :: Blob, gradient :: Blob)
  gradient.data .*= (output.data .* (1-output.data))
end
