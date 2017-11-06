export save_network, load_network

############################################################
# General IO utilities
############################################################
function mkdir_p(path::AbstractString)
  # I bet $5 this bad recursion is not a problem for our usages here
  if isempty(path) || isdir(path)
    return
  end

  if !isdir(dirname(path))
    mkdir_p(dirname(path))
  end
  mkdir(path)
end

# add this function since the built-in tempname() function is not
# working properly on Windows. See https://github.com/JuliaLang/julia/issues/9053
function temp_filename()
  tmp_dir = tempdir()
  joinpath(tmp_dir, "Mocha-$(getpid())-$(randstring(32))")
end

function glob(path::AbstractString, pattern::Regex; sort_by :: Symbol = :none)
  list = filter(x -> ismatch(pattern, x), readdir(path))
  if sort_by == :none
    return list
  elseif sort_by == :name
    return sort(list)
  elseif sort_by == :mtime
    return sort(list, by = fn -> begin
      stat(joinpath(path, fn)).mtime
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
  params_all = Dict{AbstractString, Vector{Array}}()
  for i = 1:length(net.layers)
    if has_param(net.layers[i])
      key = net.layers[i].name
      if haskey(params_all, key)
        error("Duplicated names ($key) for multiple layers, cannot save parameters")
      end

      m_debug("Saving parameters for layer $(key)")
      params = map(param -> to_array(param.blob), net.states[i].parameters)
      params_all[key] = params
    end
  end
  write(file, NETWORK_SAVE_NAME, params_all)
end

function is_similar_shape(x, y)
  function compact_shape(obj)
    shape = [size(obj)...]
    mask  = ones(Bool, length(shape))
    # strip singleton dimension at the two end
    for i = 1:length(shape)
      if shape[i] == 1
        mask[i] = false
      else
        break
      end
    end
    for i = length(shape):-1:1
      if shape[i] == 1
        mask[i] = false
      else
        break
      end
    end
    return shape[mask]
  end

  compact_shape(x) == compact_shape(y)
end

function load_network(file::JLD.JldFile, net)
  params_all = read(file, NETWORK_SAVE_NAME)
  for i = 1:length(net.layers)
    if has_param(net.layers[i])
      key = net.layers[i].name
      if !haskey(params_all, key)
        error("Cannot find saved parameters for layer $key")
      end

      m_debug("Loading parameters for layer $(key)")
      params = params_all[key]
      @assert length(params) == length(net.states[i].parameters)

      for j = 1:length(params)
        @assert is_similar_shape(params[j], net.states[i].parameters[j].blob)
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
function load_network(file::HDF5File, net, die_if_not_found=true)
  for i = 1:length(net.layers)
    if has_param(net.layers[i])
      layer_name = net.layers[i].name
      m_debug("Loading parameters from HDF5 for layer $layer_name")
      for j = 1:length(net.states[i].parameters)
        param_obj  = net.states[i].parameters[j]
        param_name = param_obj.name
        key = "$(layer_name)___$(param_name)"
        if !has(file, key)
          if param_name == "bias"
            m_warn("No bias found for $layer_name, use default initialization")
            init(param_obj.initializer, param_obj.blob)
          else
            if die_if_not_found
              error("No saved parameter $param_name not found for layer $layer_name")
            else
              # show error message but do not die
              m_warn("No saved parameter $param_name not found for layer $layer_name, use default initialization")
              init(param_obj.initializer, param_obj.blob)
            end
          end
        else
          param = read(file, key)
          if !is_similar_shape(param, param_obj.blob)
            error("Dimension for $param_name not match: got $(size(param)), expect $(size(param_obj.blob))")
          end

          if eltype(param) != eltype(param_obj.blob)
            m_warn("Automatic converting saved $param_name from $(eltype(param)) to $(eltype(param_obj.blob))")
            param = convert(Array{eltype(param_obj.blob)}, param)
          end
          copy!(param_obj.blob, param)
        end
        param_obj.initializer = NullInitializer()
      end
    end
  end
end
