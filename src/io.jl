export save_network, load_network

############################################################
# General IO utilities
############################################################
function mkdir_p(path::String)
  # I bet $5 this bad recursion is not a problem for our usages here
  if !isdir(basename(path))
    mkdir_p(basename(path))
  end
  mkdir(path)
end

function glob(path::String, pattern::Regex; sort_by :: Symbol = :none)
  list = filter(x -> ismatch(pattern, x), readdir(path))
  if sort_by == :none
    return list
  elseif sort_by == :name
    return sort(list)
  elseif sort_by == :mtime
    return sort(list, by = fn -> begin
      stat(fn).mtime
    end)
  else
    error("Unknown sort_by: $sort_by")
  end
end

############################################################
# JLD IO procedures to save and load network parameters
############################################################
const NETWORK_SAVE_NAME = "params_all"

function save_network(file, net)
  params_all = Dict{String, Vector{Array}}()
  for i = 1:length(net.layers)
    if :parameters ∈ names(net.states[i])
      key = net.layers[i].name
      if haskey(params_all, key)
        error("Duplicated names ($key) for multiple layers, cannot save parameters")
      end

      @debug("Saving parameters for layer $(key)")
      params = map(get_param_data, net.states[i].parameters)
      params_all[key] = params
    end
  end
  file[NETWORK_SAVE_NAME] = params_all
end

function get_param_data(param)
  blob = param.blob
  data = Array(eltype(blob), size(blob))
  copy!(data, blob)
  return data
end

function load_network(file, net)
  params_all = file[NETWORK_SAVE_NAME]
  for i = 1:length(net.layers)
    if :parameters ∈ names(net.states[i])
      key = net.layers[i].name
      if !haskey(params_all, key)
        error("Cannot find saved parameters for layer $key")
      end

      @debug("Loading parameters for layer $(key)")
      params = params_all[key]
      @assert length(params) == length(net.states[i].parameters)

      for j = 1:length(params)
        @assert size(params[j]) == size(net.states[i].parameters[j])
        copy!(net.states[i].parameters[j].blob, params[j])
        net.states[i].parameters[j].initializer = NullInitializer()
      end
    end
  end
end
