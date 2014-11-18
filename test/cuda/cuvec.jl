function test_cuvec(sys::System, T)
  println("-- Testing CuVec Utilities{$T}")
  width, height, channels, num = (5,6,7,8)
  spatial_dim = width*height
  dims = (width, height, channels, num)
  eps = 1e-5

  X = rand(T, dims)
  Y = rand(T, dims)
  X_blob = make_blob(sys.backend, X)
  Y_blob = make_blob(sys.backend, Y)

  println("    > mul!")
  Vec.mul!(X, Y)
  CuVec.mul!(sys, T, X_blob.ptr.p, Y_blob.ptr.p, spatial_dim, channels, num)
  X2 = similar(X)
  copy!(X2, X_blob)
  @test all(abs(X-X2) .< eps)

  println("    > pow!")
  Vec.pow!(X, 2)
  CuVec.pow!(sys, T, X_blob.ptr.p, 2, spatial_dim, channels, num)
  copy!(X2, X_blob)
  @test all(abs(X-X2) .< eps)

  Vec.pow!(X, convert(T, 0.75))
  CuVec.pow!(sys, T, X_blob.ptr.p, convert(T, 0.75), spatial_dim, channels, num)
  copy!(X2, X_blob)
  @test all(abs(X-X2) .< eps)
end

function test_cuvec(sys::System)
  test_cuvec(sys, Float32)
  test_cuvec(sys, Float64)
end

test_cuvec(sys_cudnn)
