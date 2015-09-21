############################################################
# Power Layer
############################################################
@defstruct PowerLayer Layer (
  name :: AbstractString = "power",
  (power :: Number = 1, isreal(power)),
  (scale :: Number = 1, isreal(scale)),
  (shift :: Number = 0, isreal(shift)),
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
)
@characterize_layer(PowerLayer,
  can_do_bp => true
)

type PowerLayerState <: LayerState
  layer      :: PowerLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::PowerLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  blobs = Blob[make_blob(backend, eltype(x), size(x)) for x in inputs]
  blobs_diff = Array(Blob, length(inputs))
  for i = 1:length(inputs)
    # if the bottom layer does not need back propagate, I don't need, either
    if isa(diffs[i], NullBlob)
      blobs_diff[i] = NullBlob()
    else
      blobs_diff[i] = make_blob(backend, eltype(inputs[i]), size(inputs[i]))
    end
  end
  state = PowerLayerState(layer, blobs, blobs_diff)
end
function shutdown(backend::Backend, state::PowerLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
end

function forward(backend::CPUBackend, state::PowerLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    # output = input
    copy!(output, input)

    # output *= scale
    if state.layer.scale != 1
      BLAS.scal!(length(output), convert(eltype(output),state.layer.scale), output.data, 1)
    end

    if state.layer.shift != 0
      # output += shift
      Vec.add_scal!(output.data, state.layer.shift)
    end

    # output = output ^ power
    if state.layer.power != 1
      if state.layer.power == 2
        Vec.mul!(output.data, output.data)
      else
        Vec.pow!(output.data, state.layer.power)
      end
    end
  end
end

function backward(backend::CPUBackend, state::PowerLayerState,
    inputs::Vector{Blob}, diffs::Vector{Blob})

  data_type = eltype(inputs[1])
  pow_scale = convert(data_type,state.layer.power * state.layer.scale)
  for i = 1:length(inputs)
    diff = diffs[i]
    if isa(diff, NullBlob)
      continue
    end
    if state.layer.power == 1 || state.layer.scale == 0
      # trivial case, derivative is constant
      fill!(diff, pow_scale)
    else
      input = inputs[i]
      output = state.blobs[i]

      erase!(diff)

      if state.layer.power == 2
        # dO/dI = 2 * scale * (scale * I + shift)
        #       = pow_scale * scale * I + pow_scale * shift
        BLAS.axpy!(length(input), convert(data_type, pow_scale*state.layer.scale),
            input.data, 1, diff.data, 1)
        if state.layer.shift != 0
          Vec.add_scal!(diff.data, pow_scale * state.layer.shift)
        end
      elseif state.layer.shift == 0
        # dO/dI = power * scale * (scale * I) ^ (power - 1)
        #       = power * O / I
        BLAS.axpy!(length(input), convert(data_type,state.layer.power),
            output.data, 1, diff.data, 1)
        Vec.div!(diff.data, input.data)
      else
        # general case
        # dO/dI = power * scale * (scale * I + shift) ^ (power - 1)
        #       = power * scale * O / (scale * I + shift)
        copy!(diff, input)
        if state.layer.scale != 1
          BLAS.scal!(length(diff), convert(data_type,state.layer.scale), diff.data, 1)
        end
        Vec.add_scal!(diff.data, state.layer.shift)
        Vec.div2!(output.data, diff.data)
        BLAS.scal!(length(diff), pow_scale, diff.data, 1)
      end
    end
    Vec.mul!(diff.data, state.blobs_diff[i].data)
  end
end

