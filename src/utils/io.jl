export save_network, load_network

############################################################
# General IO utilities
############################################################
function mkdir_p(path::String)
  # I bet $5 this bad recursion is not a problem for our usages here
  if isempty(path) || isdir(path)
    return
  end

  if !isdir(dirname(path))
    mkdir_p(dirname(path))
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
using HDF5, JLD
const NETWORK_SAVE_NAME = "params_all"

function save_network(file::JLD.JldFile, net)
  params_all = Dict{String, Vector{Array}}()
  for i = 1:length(net.layers)
    if isa(net.layers[i], TrainableLayer)
      key = net.layers[i].name
      if haskey(params_all, key)
        error("Duplicated names ($key) for multiple layers, cannot save parameters")
      end

      @debug("Saving parameters for layer $(key)")
      params = map(get_param_data, net.states[i].parameters)
      params_all[key] = params
    end
  end
  write(file, NETWORK_SAVE_NAME, params_all)
end

function get_param_data(param)
  blob = param.blob
  data = Array(eltype(blob), size(blob))
  copy!(data, blob)
  return data
end

function load_network(file::JLD.JldFile, net)
  params_all = read(file, NETWORK_SAVE_NAME)
  for i = 1:length(net.layers)
    if isa(net.layers[i], TrainableLayer)
      key = net.layers[i].name
      if !haskey(params_all, key)
        error("Cannot find saved parameters for layer $key")
      end

      @debug("Loading parameters for layer $(key)")
      params = params_all[key]
      @assert length(params) == length(net.states[i].parameters)

      for j = 1:length(params)
        @assert size(params[j]) == size(net.states[i].parameters[j].blob)
        copy!(net.states[i].parameters[j].blob, params[j])
        net.states[i].parameters[j].initializer = NullInitializer()
      end
    end
  end
end

############################################################
# Load network parameters from a HDF5 file
# When mapped correctly, this could be used to import
# trained network from other tools (e.g. Caffe)
############################################################
function load_network(file::HDF5File, net)
  for i = 1:length(net.layers)
    if isa(net.layers[i], TrainableLayer)
      layer_name = net.layers[i].name
      @debug("Loading parameters from HDF5 for layer $layer_name")
      for j = 1:length(net.states[i].parameters)
        param_obj  = net.states[i].parameters[j]
        param_name = param_obj.name
        key = "$layer_name___$param_name"
        if !has(file, key)
          if param_name == "bias"
            @warn("No bias found for $layer_name, initializing as zeros")
            fill!(param_obj.blob, 0)
          else
            @error("No saved parameter $param_name not found for layer $layer_name")
          end
        else
          param = read(file, param_name)
          @assert size(param) == size(param_obj.blob) "Dimension for saved parameter $param_name does not match"
          if eltype(param) != eltype(param_obj.blob)
            @warn("Automatic converting saved $param_name from $(eltype(param)) to $(eltype(param_obj.blob))")
            param = convert(Array{eltype(param_obj.blob)}, param)
          end
          copy!(param_obj.blob, param)
        end
        param_obj.initializer = NullInitializer()
      end
    end
  end
end
