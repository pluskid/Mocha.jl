function test_raw_blas()
  println("-- Testing RawBLAS Utilities")

  eps = 1e-10
  M, N, K = (108, 6, 36)
  A = rand(M, K)
  B = rand(N, K)

  C = rand(M, N)
  C2 = rand(M, N)
  RawBLAS.gemm!('N', 'T', M, N, K, 1.0, A, M, B, N, 0.0, C, M)
  BLAS.gemm!('N', 'T', 1.0, A, B, 0.0, C2)

  @test all(abs(C - C2) .< eps)

  A2 = rand(M, 2K)
  B2 = rand(N, 3K)
  C = rand(M, N)
  C2 = A2[:, 1:K] * B2[:, 1:K]'
  RawBLAS.gemm!('N', 'T', M, N, K, 1.0, A2, M, B2, N, 0.0, C, M)
  @test all(abs(C - C2) .< eps)

  C2 = A2[:, K+1:2K] * B2[:, 2K+1:3K]'
  C = rand(M, 2N)
  RawBLAS.gemm!('N', 'T', M, N, K, 1.0,
      convert(Ptr{Float64},A2) + M*K*sizeof(Float64), M,
      convert(Ptr{Float64},B2) + 2N*K*sizeof(Float64), N,
      0.0, convert(Ptr{Float64},C) + M*N*sizeof(Float64), M)
  @test all(abs(C2 - C[1:M,N+1:2N]) .< eps)
end

test_raw_blas()
