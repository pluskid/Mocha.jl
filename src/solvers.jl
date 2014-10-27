export Solver
export SGD

export solve

abstract Solver

############################################################
# General utilities that could be used by all solvers
############################################################
# Initialize network parameters according to defined initializers
function init(net::Net)
  for i = 1:length(net.layers)
    state = net.states[i]
    if :parameters ∈ names(state)
      for j = 1:length(state.parameters)
        init(state.parameters[j].initializer, state.parameters[j].blob)
      end
    end
  end
end

include("solvers/sgd.jl")
