function forward(backend::GPUBackend, state::ArgmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    spatial_dim, channels, num = split_dims(input, state.dims[i])
    data_type = eltype(input)

    x_block = round(Int64, ceil(float64(num)/CUDA.THREADS_PER_BLOCK_X));
    y_block = round(Int64, ceil(float64(spatial_dim)/CUDA.THREADS_PER_BLOCK_Y));

    if data_type == Float32
      kernel = backend.mocha.argmax_forward_float
    elseif data_type == Float64
      kernel = backend.mocha.argmax_forward_double
    else
      error("Unsupported data type $data_type")
    end

    CUDA.launch(kernel, (x_block,y_block),(CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y),
        (input.ptr.p, output.ptr.p, num, channels, spatial_dim));
  end
end
