function test_crop_layer(backend::Backend, do_mirror, n_input, T, eps)
  println("-- Testing CropLayer on $(typeof(backend)){$T} $(do_mirror ? "with mirror" : "")...")

  dims = [rand(7:13, 4) for i = 1:n_input]
  input = [rand(T, dims[i]...) for i = 1:n_input]
  crop_size = (7,5)
  input_blob = Blob[make_blob(backend, x) for x in input]
  diff_blob = Blob[NullBlob() for i = 1:n_input]

  println("    > Setup")
  layer = CropLayer(bottoms=Array{Symbol}(n_input), tops=Array{Symbol}(n_input),
      crop_size=crop_size, random_mirror=do_mirror)
  state = setup(backend, layer, input_blob, diff_blob)

  println("    > Forward")
  forward(backend, state, input_blob)
  for i = 1:n_input
    got_output = zeros(T, size(state.blobs[i]))
    copy!(got_output, state.blobs[i])

    w_off = div(size(input[i],1)-crop_size[1],2)
    h_off = div(size(input[i],2)-crop_size[2],2)
    expected_output = input[i][w_off+1:w_off+crop_size[1], h_off+1:h_off+crop_size[2],:,:]
    @test size(expected_output) == size(got_output)
    if do_mirror
      @test all(abs.(got_output - expected_output) .< eps) ||
            all(abs.(got_output - flipdim(expected_output,1)) .< eps)
    else
      @test all(abs.(got_output - expected_output) .< eps)
    end
  end

  shutdown(backend, state)
end
function test_crop_layer_random(backend::Backend, do_mirror, n_input, T, eps)
  println("-- Testing CropLayer{rnd} on $(typeof(backend)){$T} $(do_mirror ? "with mirror" : "")...")
  dims = [rand(9:11, 4) for i = 1:n_input]
  inputs = [rand(T, dims[i]...) for i = 1:n_input]
  crop_size = (9, 9)
  input_blob = Blob[make_blob(backend, x) for x in inputs]
  diff_blob = Blob[NullBlob() for i = 1:n_input]

  println("    > Setup")
  layer = CropLayer(bottoms=Array{Symbol}(n_input), tops=Array{Symbol}(n_input), crop_size=crop_size,
      random_mirror=do_mirror, random_crop=true)
  state = setup(backend, layer, input_blob, diff_blob)

  println("    > Forward")
  forward(backend, state, input_blob)
  for kk = 1:n_input
    got_output = zeros(T, size(state.blobs[kk]))
    copy!(got_output, state.blobs[kk])
    input = inputs[kk]

    matched = false
    for i = 0:size(input,1)-crop_size[1]
      for j = 0:size(input,2)-crop_size[2]
        expected_output = input[i+1:i+crop_size[1], j+1:j+crop_size[2],:,:]
        matched = matched | all(abs.(got_output - expected_output) .< eps)
        if do_mirror
          matched = matched | all(abs.(got_output - flipdim(expected_output,1)) .< eps)
        end
      end
    end
    @test matched
  end

  shutdown(backend, state)
end

function test_crop_layer(backend::Backend, n_input, T, eps)
  test_crop_layer(backend, false, n_input, T, eps)
  test_crop_layer(backend, true, n_input, T, eps)
  test_crop_layer_random(backend, false, n_input, T, eps)
  test_crop_layer_random(backend, true, n_input, T, eps)
end

function test_crop_layer(backend::Backend)
  test_crop_layer(backend, 3, Float64, 1e-10)
  test_crop_layer(backend, 3, Float32, 1e-4)
end

if test_cpu
  test_crop_layer(backend_cpu)
end
if test_gpu
  test_crop_layer(backend_gpu)
end
