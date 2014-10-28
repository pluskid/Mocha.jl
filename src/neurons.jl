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
function forward(sys :: System{CPU}, neuron :: Neurons.Identity, output :: Blob)
  # do nothing
end
function backward(sys :: System{CPU}, neuron :: Neurons.Identity, output :: Blob, gradient :: Blob)
  # do nothing
end

############################################################
# Rectified-Linear
############################################################
function forward(sys :: System{CPU}, neuron :: Neurons.ReLU, output :: Blob)
  output.data[:] = max(output.data[:], 0)
end
function backward(sys :: System{CPU}, neuron :: Neurons.ReLU, output :: Blob, gradient :: Blob)
  gradient.data[:] .*= (output.data[:] .>= 0)
end

############################################################
# Sigmoid
############################################################
function forward(sys :: System{CPU}, neuron :: Neurons.Sigmoid, output :: Blob)
  output.data[:] = 1 ./ (1 + exp(-output.data[:]))
end
function backward(sys :: System{CPU}, neuron :: Neurons.Sigmoid, output :: Blob, gradient :: Blob)
  gradient.data[:] .*= (output.data[:] .* (1-output.data[:]))
end
