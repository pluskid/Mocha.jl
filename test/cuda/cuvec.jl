function test_cuvec(backend::Backend, T)
  println("-- Testing CuVec Utilities{$T}")
  dims = (5,6,7,8)
  len = prod(dims)
  eps = 1e-5

  X = rand(T, dims)
  Y = rand(T, dims)
  X_blob = make_blob(backend, X)
  Y_blob = make_blob(backend, Y)

  println("    > mul!")
  Vec.mul!(X, Y)
  CuVec.mul!(backend, T, X_blob.ptr.p, Y_blob.ptr.p, len)
  X2 = similar(X)
  copy!(X2, X_blob)
  @test all(abs(X-X2) .< eps)

  println("    > pow!")
  Vec.pow!(X, 2)
  CuVec.pow!(backend, T, X_blob.ptr.p, 2, len)
  copy!(X2, X_blob)
  @test all(abs(X-X2) .< eps)

  Vec.pow!(X, convert(T, 0.75))
  CuVec.pow!(backend, T, X_blob.ptr.p, convert(T, 0.75), len)
  copy!(X2, X_blob)
  @test all(abs(X-X2) .< eps)

  println("    > log!")
  X = max(X, convert(T,1e-20))
  copy!(X_blob, X)
  CuVec.log!(backend, X_blob)
  X2 = to_array(X_blob)
  @test all(abs(X2 - log(X)) .< eps)
end

function test_cuvec(backend::Backend)
  test_cuvec(backend, Float32)
  test_cuvec(backend, Float64)
end

test_cuvec(backend_gpu)
