export RawBLAS

module RawBLAS
using Base.LinAlg # force built-in BLAS library initialization
using ..Mocha.blasfunc
using Compat

for (gemm, elty) in ((:dgemm_, Float64), (:sgemm_, Float32))
  @eval begin
    function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int, alpha::$elty,
        A, lda, B, ldb, beta::$elty, C, ldc)
      ccall(($(blasfunc(gemm)), Base.libblas_name), Void,
          (Ptr{UInt8}, Ptr{UInt8}, Ptr{Base.LinAlg.BlasInt}, Ptr{Base.LinAlg.BlasInt},
          Ptr{Base.LinAlg.BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{Base.LinAlg.BlasInt},
          Ptr{$elty}, Ptr{Base.LinAlg.BlasInt}, Ptr{$elty}, Ptr{$elty},
          Ptr{Base.LinAlg.BlasInt}),
          &transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)
    end

    function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int, alpha::$elty,
        A, B, beta::$elty, C)
      lda = (transA == 'N') ? M : K;
      ldb = (transB == 'N') ? K : N;
      ldc = M;
      gemm!(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    end
  end # eval
end

for (gemv, elty) in ((:dgemv_, Float64), (:sgemv_, Float32))
  @eval begin
    function gemv!(trans::Char, M::Int, N::Int, alpha::$elty, A, lda, x, incx, beta::$elty, y, incy)
      ccall(($(blasfunc(gemv)), Base.libblas_name), Void,
          (Ptr{UInt8}, Ptr{Base.LinAlg.BlasInt}, Ptr{Base.LinAlg.BlasInt}, Ptr{$elty},
          Ptr{$elty}, Ptr{Base.LinAlg.BlasInt}, Ptr{$elty}, Ptr{Base.LinAlg.BlasInt},
          Ptr{$elty}, Ptr{$elty}, Ptr{Base.LinAlg.BlasInt}),
          &trans, &M, &N, &alpha, A, &lda, x, &incx, &beta, y, &incy)
    end
    function gemv!(trans::Char, M::Int, N::Int, alpha::$elty, A, x, beta::$elty, y)
      lda = M
      incx = 1
      incy = 1
      gemv!(trans, M, N, alpha, A, lda, x, incx, beta, y, incy)
    end
  end
end

end # module RawBLAS
