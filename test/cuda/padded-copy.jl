function test_padded_copy(sys::System, T)
  println("-- Testing CUDA PaddedCopy Utilities{$T}")
  eps = 1e-10
  width, height, channels, num = (4,5,6,7)
  pad = (2,3)

  orig = rand(T, width, height, channels, num)
  padded = round(100*rand(T, width+2*pad[1], height+2*pad[2], channels, num))

  orig_blob = make_blob(sys.backend, orig)
  padded_blob = make_blob(sys.backend, padded)
  dense2padded!(sys, padded_blob, orig_blob, pad)
  copy!(padded, padded_blob)

  got = padded[pad[1]+1:end-pad[1],pad[2]+1:end-pad[2],:,:]

  @test all(abs(got - orig) .< eps)

  padded = rand(T, size(padded))
  copy!(padded_blob, padded)
  padded2dense!(sys, orig_blob, padded_blob, pad)
  copy!(orig, orig_blob)

  expected = padded[pad[1]+1:end-pad[1],pad[2]+1:end-pad[2],:,:]
  @test all(abs(expected - orig) .< eps)
end

function test_padded_copy(sys::System)
  test_padded_copy(sys, Float32)
  test_padded_copy(sys, Float64)
end

test_padded_copy(sys_cudnn)

