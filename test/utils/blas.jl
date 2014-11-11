function test_raw_blas()
  println("-- Testing RawBLAS Utilities")

  eps = 1e-10
  M, N, K = (8, 9, 10)
  A = rand(M, K)
  B = rand(N, K)

  C = rand(M, N)
  RawBLAS.gemm!('N', 'T', M, N, K, 1.0, A, B, 0.0, C)

  @test all(abs(C - A*B') .< eps)
end

test_raw_blas()
