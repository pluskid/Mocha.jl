export Net
export init, destroy, forward, backward, forward_backward, get_epoch
export show_statistics, reset_statistics

type Net{T <: Backend}
  sys :: System{T}

  # all layers, sorted in topological order
  layers :: Vector{Layer}

  states         :: Vector{LayerState}
  blobs_forward  :: Vector{Vector{Blob}}
  blobs_backward :: Vector{Vector{Blob}}
  data_layers    :: Vector{Int}
end

function get_epoch(net::Net)
  if length(net.data_layers) == 0
    error("No data layer in the net, cannot get epoch")
  end
  return net.states[net.data_layers[1]].epoch
end

function init(net::Net, regu_coef :: FloatingPoint = 0.0)
  for i = 1:length(net.layers)
    state = net.states[i]
    if :parameters ∈ names(state)
      for param in state.parameters
        init(param.initializer, param.blob)

        # scale per-layer regularization coefficient globally
        param.regularizer.coefficient *= regu_coef
      end
    end
  end
end
function destroy(net::Net)
  # TODO
end

function show_statistics(net::Net; title="Network Statistics")
  @info("")
  @info("## $title")
  @info("---------------------------------------------------------")
  for i = 1:length(net.layers)
    if isa(net.layers[i], StatLayer)
      show_statistics(net.states[i])
    end
  end
  @info("---------------------------------------------------------")
  @info("")
end
function reset_statistics(net::Net)
  for i = 1:length(net.layers)
    if isa(net.layers[i], StatLayer)
      reset_statistics(net.states[i])
    end
  end
end

function forward_backward(net::Net)
  obj_val = forward(net)
  backward(net)
  return obj_val
end

function forward(net::Net)
  obj_val = 0.0

  for i = 1:length(net.layers)
    forward(net.sys, net.states[i], net.blobs_forward[i])
    if :neuron ∈ names(net.layers[i]) && !isa(net.layers[i].neuron, Neurons.Identity)
      for blob in net.states[i].blobs
        forward(net.sys, net.layers[i].neuron, blob)
      end
    end

    if isa(net.layers[i], LossLayer)
      obj_val += net.states[i].loss
    end

    # handle regularization
    if :parameters ∈ names(net.states[i])
      for param in net.states[i].parameters
        obj_val += forward(net.sys, param.regularizer, param.blob)
      end
    end
  end

  return obj_val
end

function backward(net::Net)
  for i = length(net.layers):-1:1
    if :neuron ∈ names(net.layers[i]) && !isa(net.layers[i].neuron, Neurons.Identity)
      state = net.states[i]
      for j = 1:length(state.blobs)
        backward(net.sys, net.layers[i].neuron, state.blobs[j], state.blobs_diff[j])
      end
    end
    backward(net.sys, net.states[i], net.blobs_forward[i], net.blobs_backward[i])

    # handle regularization
    if :parameters ∈ names(net.states[i])
      for param in net.states[i].parameters
        backward(net.sys, param.regularizer, param.blob, param.gradient)
      end
    end
  end
end


Net(sys::System, layers :: Vector{Layer}) = begin
  layers = topological_sort(layers)
  data_layers = find(l -> isa(l, DataLayer), layers)

  n = length(layers)
  states = Array(LayerState, n)
  blobs_forward = Array(Vector{Blob}, n)
  blobs_backward = Array(Vector{Blob}, n)

  output_blobs = Dict{Symbol,Blob}()
  diff_blobs = Dict{Symbol,Blob}()

  for i = 1:n
    layer = layers[i]
    # record if layers has any dependency
    if :bottoms ∈ names(layer)
      blob_fwd = Blob[output_blobs[x] for x in layer.bottoms]
      blob_bwd = Blob[haskey(diff_blobs,x) ? diff_blobs[x] : NullBlob() for x in layer.bottoms]
    else
      blob_fwd = Blob[]
      blob_bwd = Blob[]
    end

    if haskey(sys.layer_registry, layers[i])
      shared_state = sys.layer_registry[layers[i]]
      states[i] = setup(sys, layers[i], shared_state, blob_fwd)

      # shared parameters, don't re-initialize
      for param in states[i].parameters
        param.initializer = NullInitializer()
      end
    else
      states[i] = setup(sys, layers[i], blob_fwd)
      if :parameters ∈ names(states[i])
        # has parameters, save in registry
        sys.layer_registry[layers[i]] = states[i]
      end
    end

    if :tops ∈ names(layer)
      for j = 1:length(layer.tops)
        output_blobs[layer.tops[j]] = states[i].blobs[j]
        if :blobs_diff ∈ names(states[i])
          diff_blobs[layer.tops[j]] = states[i].blobs_diff[j]
        end
      end
    end

    blobs_forward[i] = blob_fwd
    blobs_backward[i] = blob_bwd
  end

  return Net(sys, layers, states, blobs_forward, blobs_backward, data_layers)
end


function topological_sort(layers :: Vector{Layer})
  n = length(layers)

  #---- Build dependency graph
  graph = zeros(Int, n, n)
  outputs = Dict{Symbol, Int}()

  for i = 1:n
    if :tops ∈ names(layers[i])
      for key in layers[i].tops
        if haskey(outputs, key)
          error("Duplicated output blob name: $(key)")
        end
        outputs[key] = i
      end
    end
  end

  for i = 1:n
    if :bottoms ∈ names(layers[i])
      for key in layers[i].bottoms
        if !haskey(outputs, key)
          error("Required input blob missing: $(key)")
        end
        graph[i,outputs[key]] = 1
      end
    end
  end

  #---- Topological sort
  index = Int[]
  while length(index) < n
    # find layers that has no dependency
    idx = find(sum(graph,2) .== 0)
    if length(idx) == 0
      error("Can't finish topological sort, cycle in layer dependency?")
    end

    push!(index, idx...)
    graph[idx,:] = 2 # make sure we don't select those again
    graph[:,idx] = 0 # layers that depend on those could be selected
  end

  return layers[index]
end
