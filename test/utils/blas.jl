function test_raw_blas(T)
  println("-- Testing RawBLAS{$T} Utilities")

  eps = 1e-10
  M, N, K = (108, 6, 36)
  A = rand(T, M, K)
  B = rand(T, N, K)

  C = rand(T, M, N)
  C2 = rand(T, M, N)
  RawBLAS.gemm!('N', 'T', M, N, K, convert(T,1.0), A, M, B, N, convert(T,0.0), C, M)
  BLAS.gemm!('N', 'T', convert(T,1.0), A, B, convert(T,0.0), C2)

  @test all(abs.(C - C2) .< eps)

  A2 = rand(T, M, 2K)
  B2 = rand(T, N, 3K)
  C = rand(T, M, N)
  C2 = A2[:, 1:K] * B2[:, 1:K]'
  RawBLAS.gemm!('N', 'T', M, N, K, convert(T,1.0), A2, M, B2, N, convert(T,0.0), C, M)
  @test all(abs.(C - C2) .< eps)

  C2 = A2[:, K+1:2K] * B2[:, 2K+1:3K]'
  C = rand(T, M, 2N)
  RawBLAS.gemm!('N', 'T', M, N, K, convert(T, 1.0),
      pointer(A2) + M*K*sizeof(T), M,
      pointer(B2) + 2N*K*sizeof(T), N,
      convert(T,0.0), pointer(C) + M*N*sizeof(T), M)
  @test all(abs.(C2 - C[1:M,N+1:2N]) .< eps)
end

test_raw_blas() = begin
  test_raw_blas(Float32)
  test_raw_blas(Float64)
end

test_raw_blas()
