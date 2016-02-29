export ActivationFunction
export Neurons

export forward, backward

############################################################
# Each activation function / neuron should be a sub type
# of ActivationFunction and should implement the following
# methods.
#
# - forward(backend :: Backend, neuron :: MyActivationFunction, output :: Blob)
#   Update the output blob by the activation function. Computation should be done in-place.
#   Note we call is "output" because an activation function is applied to the "output" of
#   a layer.
#
# - backward(backend :: Backend, neuron :: MyActivationFunction, output :: Blob, gradient :: Blob)
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

# Rectified-Linear: ReLU(eps)(x) = max(x,eps)
type ReLU <: ActivationFunction
  epsilon::Float64 # optional floor value, default zero
end
ReLU() = ReLU(0.0)

# Exponential: Exponential(x) = exp(x)
type Exponential <: ActivationFunction
end

# Leaky Rectified-Linear: LReLU(x) = x > 0 ? x : 0.01x
type LReLU <: ActivationFunction
end

# Sigmoid: Sigmoid(x) = 1 / (1 + exp(-x))
type Sigmoid <: ActivationFunction
end

# Sigmoid: Tanh(x) = (1 + exp(-2x)) / (1 + exp(-2x))
type Tanh <: ActivationFunction
end
end # module Neurons

############################################################
# Identity
############################################################
function forward(backend :: Backend, neuron :: Neurons.Identity, output :: Blob)
  # do nothing
end
function backward(backend :: Backend, neuron :: Neurons.Identity, output :: Blob, gradient :: Blob)
  # do nothing
end

############################################################
# Rectified-Linear
############################################################
function forward(backend :: CPUBackend, neuron :: Neurons.ReLU, output :: Blob)
  function _forward(output::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds output[i] = max(neuron.epsilon, output[i])
    end
  end
  _forward(output.data)
end
function backward(backend :: CPUBackend, neuron :: Neurons.ReLU, output :: Blob, gradient :: Blob)
  function _backward(output::AbstractArray, gradient::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds gradient[i] *= (output[i] > neuron.epsilon)
    end
  end
  _backward(output.data, gradient.data)
end

############################################################
# Leaky Rectified-Linear
############################################################
function forward(backend :: CPUBackend, neuron :: Neurons.LReLU, output :: Blob)
  function _forward(output::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds output[i] = output[i] > 0 ? output[i] : 0.01 * output[i]
    end
  end
  _forward(output.data)
end
function backward(backend :: CPUBackend, neuron :: Neurons.LReLU, output :: Blob, gradient :: Blob)
  function _backward(output::AbstractArray, gradient::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds gradient[i] *= ((output[i] > 0) + 0.01 * (output[i] <= 0))
    end
  end
  _backward(output.data, gradient.data)
end

############################################################
# Sigmoid
############################################################
function forward(backend :: CPUBackend, neuron :: Neurons.Sigmoid, output :: Blob)
  function _forward(output::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds output[i] = 1 / (1 + exp(-output[i]))
    end
  end
  _forward(output.data)
end
function backward(backend :: CPUBackend, neuron :: Neurons.Sigmoid, output :: Blob, gradient :: Blob)
  function _backward(output::AbstractArray, gradient::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds gradient[i] *= output[i] * (1-output[i])
    end
  end
  _backward(output.data, gradient.data)
end


############################################################
# Tanh
############################################################
function forward(backend :: CPUBackend, neuron :: Neurons.Tanh, output :: Blob)
  function _forward(output::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds output[i] = tanh(output[i])
    end
  end
  _forward(output.data)
end
function backward(backend :: CPUBackend, neuron :: Neurons.Tanh, output :: Blob, gradient :: Blob)
  function _backward(output::AbstractArray, gradient::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds gradient[i] *= (1 - output[i] * output[i])
    end
  end
  _backward(output.data, gradient.data)
end



############################################################
# Exponential
############################################################
function forward(backend :: CPUBackend, neuron :: Neurons.Exponential, output :: Blob)
  function _forward(output::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds output[i] = exp(output[i])
    end
  end
  _forward(output.data)
end
function backward(backend :: CPUBackend, neuron :: Neurons.Exponential, output :: Blob, gradient :: Blob)
  function _backward(output::AbstractArray, gradient::AbstractArray)
    @simd for i = 1:length(output)
      @inbounds gradient[i] *= output[i]
    end
  end
  _backward(output.data, gradient.data)
end
