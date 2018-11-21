export RawBLAS

module RawBLAS
using LinearAlgebra # force built-in BLAS library initialization
using ..Mocha: blasfunc
using Compat

for (gemm, elty) in ((:dgemm_, Float64), (:sgemm_, Float32))
  @eval begin
    function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int, alpha::$elty,
        A, lda, B, ldb, beta::$elty, C, ldc)
      transA = convert(Cuchar, transA)
      transB = convert(Cuchar, transB)
      ccall(($(blasfunc(gemm)), Base.libblas_name), Nothing,
          (Ptr{Cuchar}, Ptr{Cuchar}, Ptr{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt},
          Ptr{LinearAlgebra.BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{LinearAlgebra.BlasInt},
          Ptr{$elty}, Ptr{LinearAlgebra.BlasInt}, Ptr{$elty}, Ptr{$elty},
          Ptr{LinearAlgebra.BlasInt}),
          Ref(transA), Ref(transB), Ref(M), Ref(N), Ref(K), Ref(alpha), A, Ref(lda), B, Ref(ldb), Ref(beta), C, Ref(ldc))
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
      trans = convert(Cuchar, trans)
      ccall(($(blasfunc(gemv)), Base.libblas_name), Nothing,
          (Ptr{UInt8}, Ptr{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}, Ptr{$elty},
          Ptr{$elty}, Ptr{LinearAlgebra.BlasInt}, Ptr{$elty}, Ptr{LinearAlgebra.BlasInt},
          Ptr{$elty}, Ptr{$elty}, Ptr{LinearAlgebra.BlasInt}),
          Ref(trans), Ref(M), Ref(N), Ref(alpha), A, Ref(lda), x, Ref(incx), Ref(beta), y, Ref(incy))
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
