function test_crop_layer(sys::System, do_mirror, T, eps)
  println("-- Testing CropLayer on $(typeof(sys.backend)){$T} $(do_mirror ? "with mirror" : "")...")

  input = rand(T, 11, 12, 5, 6)
  crop_size = (7,5)
  input_blob = make_blob(sys.backend, input)

  println("    > Setup")
  layer = CropLayer(bottoms=[:input], tops=[:output], crop_size=crop_size, random_mirror=do_mirror)
  state = setup(sys, layer, Blob[input_blob], Blob[NullBlob()])

  println("    > Forward")
  forward(sys, state, Blob[input_blob])
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

  shutdown(sys, state)
end
function test_crop_layer(sys::System, T, eps)
  test_crop_layer(sys, false, T, eps)
  test_crop_layer(sys, true, T, eps)
end

function test_crop_layer(sys::System)
  test_crop_layer(sys, Float64, 1e-10)
  test_crop_layer(sys, Float32, 1e-4)
end

if test_cpu
  test_crop_layer(sys_cpu)
end
if test_cudnn
  test_crop_layer(sys_cudnn)
end

