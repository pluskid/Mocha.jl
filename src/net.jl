type Net
  # all layers, sorted in topological order
  layers :: Vector{Layer}

  states         :: Vector{LayerState}
  blobs_forward  :: Vector{Vector{Blob}}
  blobs_backward :: Vector{Vector{Blob}}

  Net(layers :: Vector{Layer}) = begin
    layers = topological_sort(layers)
    net = new(layers)

    n = length(layers)
    states = Array(LayerState, n)
    blobs_forward = Array(Vector{Blob}, n)
    blobs_backward = Array(Vector{Blob}, n)

    output_blobs = Dict{String,Blob}()
    diff_blobs = Dict{String,Blob}()

    for i = 1:n
      layer = layers[i]
      if in(:bottoms, layer)
        blob_fwd = Blob[output_blobs[x] for x in layer.bottoms]
        blob_bwd = Blob[haskey(diff_blobs,x) ? diff_blobs[x] : NullBlob() for x in layer.bottoms]
      else
        blob_fwd = Blob[]
        blob_bwd = Blob[]
      end

      states[i] = setup(layers[i], blob_fwd)
      for j = 1:length(layer.tops)
        output_blobs[layer.tops[j]] = states[i].blobs[j]
        if in(:blobs_diff, states[i])
          diff_blobs[layer.tops[j]] = states[i].blobs_diff[j]
        end
      end
      blobs_forward[i] = blob_fwd
      blobs_backward[i] = blob_bwd
    end

    net.states = states
    net.blobs_forward = blobs_forward
    net.blobs_backward = blobs_backward

    return net
  end
end

function topological_sort(layers :: Vector{Layer})
  n = length(layers)

  #---- Build dependency graph
  graph = zeros(Int, n, n)
  outputs = Dict{String, Int}()

  for i = 1:n
    for key in layers[i].tops
      if haskey(outputs, key)
        error("Duplicated output blob name: $(key)")
      end
      outputs[key] = i
    end
  end

  for i = 1:n
    if in(:bottoms, names(layers[i]))
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
