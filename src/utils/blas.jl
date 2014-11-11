export RawBLAS

module RawBLAS

for (gemm, elty) in ((:dgemm_, Float64), (:sgemm_, Float32))
  @eval begin
    function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int, alpha::$elty, A, B, beta::$elty, C)
      lda = (transA == 'N') ? M : K
      ldb = (transB == 'N') ? K : N
      ldc = M

      ccall(($(string(gemm)), Base.libblas_name), Void,
          (Ptr{Uint8}, Ptr{Uint8}, Ptr{Cint}, Ptr{Cint},
          Ptr{Cint}, Ptr{$elty}, Ptr{$elty}, Ptr{Cint},
          Ptr{$elty}, Ptr{Cint}, Ptr{$elty}, Ptr{$elty},
          Ptr{Cint}),
          &transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)
    end
  end # eval
end

end # module RawBLAS
