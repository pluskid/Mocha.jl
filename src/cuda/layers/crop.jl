function forward(backend::GPUBackend, state::CropLayerState, inputs::Vector{Blob})
  crop_size = state.layer.crop_size
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]
    if state.layer.random_crop
      w_off = abs(rand(Int)) % (size(input,1)-crop_size[1]+1)
      h_off = abs(rand(Int)) % (size(input,2)-crop_size[2]+1)
    else
      w_off = div(size(input,1)-crop_size[1], 2)
      h_off = div(size(input,2)-crop_size[2], 2)
    end
    w_off2 = size(input,1) - crop_size[1] - w_off
    h_off2 = size(input,2) - crop_size[2] - h_off

    if state.layer.random_mirror && rand(Uint)%2 == 0
      mirror = true;
    else
      mirror = false;
    end
    padded2dense!(backend, output, input, (w_off, h_off), (w_off2, h_off2), mirror)
  end
end

