function test_crop_layer(backend::Backend, do_mirror, T, eps)
  println("-- Testing CropLayer on $(typeof(backend)){$T} $(do_mirror ? "with mirror" : "")...")

  input = rand(T, 11, 12, 5, 6)
  crop_size = (7,5)
  input_blob = make_blob(backend, input)

  println("    > Setup")
  layer = CropLayer(bottoms=[:input], tops=[:output], crop_size=crop_size, random_mirror=do_mirror)
  state = setup(backend, layer, Blob[input_blob], Blob[NullBlob()])

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = zeros(T, size(state.blobs[1]))
  copy!(got_output, state.blobs[1])

  w_off = div(size(input,1)-crop_size[1],2)
  h_off = div(size(input,2)-crop_size[2],2)
  expected_output = input[w_off+1:w_off+crop_size[1], h_off+1:h_off+crop_size[2],:,:]
  @test size(expected_output) == size(got_output)
  if do_mirror
    @test all(abs(got_output - expected_output) .< eps) ||
          all(abs(got_output - flipdim(expected_output,1)) .< eps)
  else
    @test all(abs(got_output - expected_output) .< eps)
  end

  shutdown(backend, state)
end
function test_crop_layer_random(backend::Backend, do_mirror, T, eps)
  println("-- Testing CropLayer{rnd} on $(typeof(backend)){$T} $(do_mirror ? "with mirror" : "")...")
  input = rand(T, 11, 12, 5, 6)
  crop_size = (9, 9)
  input_blob = make_blob(backend, input)

  println("    > Setup")
  layer = CropLayer(bottoms=[:input], tops=[:output], crop_size=crop_size,
      random_mirror=do_mirror, random_crop=true)
  state = setup(backend, layer, Blob[input_blob], Blob[NullBlob()])

  println("    > Forward")
  forward(backend, state, Blob[input_blob])
  got_output = zeros(T, size(state.blobs[1]))
  copy!(got_output, state.blobs[1])

  matched = false
  for i = 0:size(input,1)-crop_size[1]
    for j = 0:size(input,2)-crop_size[2]
      expected_output = input[i+1:i+crop_size[1], j+1:j+crop_size[2],:,:]
      matched = matched | all(abs(got_output - expected_output) .< eps)
      if do_mirror
        matched = matched | all(abs(got_output - flipdim(expected_output,1)) .< eps)
      end
    end
  end
  @test matched

  shutdown(backend, state)
end

function test_crop_layer(backend::Backend, T, eps)
  test_crop_layer(backend, false, T, eps)
  test_crop_layer(backend, true, T, eps)
  test_crop_layer_random(backend, false, T, eps)
  test_crop_layer_random(backend, true, T, eps)
end

function test_crop_layer(backend::Backend)
  test_crop_layer(backend, Float64, 1e-10)
  test_crop_layer(backend, Float32, 1e-4)
end

if test_cpu
  test_crop_layer(backend_cpu)
end
if test_cudnn
  test_crop_layer(backend_cudnn)
end
