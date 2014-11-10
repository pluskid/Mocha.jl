export Net
export get_epoch

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
  return net.states[layer.data_layers[1]].epoch
end

Net(sys::System, layers :: Vector{Layer}) = begin
  layers = topological_sort(layers)
  data_layers = find(l -> isa(l, DataLayer), layers)

  n = length(layers)
  states = Array(LayerState, n)
  blobs_forward = Array(Vector{Blob}, n)
  blobs_backward = Array(Vector{Blob}, n)

  output_blobs = Dict{String,Blob}()
  diff_blobs = Dict{String,Blob}()

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
      @debug("Sharing parameter with existing layer $(layers[i])")
      states[i] = sys.layer_registry[layers[i]]
    else
      states[i] = setup(sys, layers[i], blob_fwd)
      sys.layer_registry[layers[i]] = states[i]
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
  outputs = Dict{String, Int}()

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
