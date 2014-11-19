function test_cublas(sys::System, T)
  println("-- Testing CuBLAS Utilities on $T")
  dims = (2,3,5,7)
  eps = 1e-5

  println("    > memory copy between device and host")
  x = rand(T, dims)
  x2 = similar(x)
  x_blob = Mocha.cudnn_make_tensor_blob(T, dims...)
  x2_blob = Mocha.cudnn_make_tensor_blob(T, dims...)
  copy!(x_blob, x)
  copy!(x2, x_blob)
  @test all(-eps .< x2-x .< eps)

  println("    > memory copy between devices")
  fill!(x2, 0)
  copy!(x2_blob, x_blob)
  copy!(x2, x2_blob)
  @test all(-eps .< x2-x .< eps)

  println("    > axpy")
  CuBLAS.axpy(sys.backend.cublas_ctx, length(x), convert(T,2.0), x_blob.ptr, 1, x2_blob.ptr, 1)
  copy!(x2, x2_blob)
  @test all(-eps .< 3*x - x2 .< eps)

  println("    > scal")
  copy!(x_blob, x)
  CuBLAS.scal(sys.backend.cublas_ctx, length(x), convert(T,0.9), x_blob.ptr, 1)
  copy!(x2, x_blob)
  @test all(-eps .< 0.9*x - x2 .< eps)

  println("    > gemm")
  A = rand(T, 5,6)
  B = rand(T, 6,7)
  C = rand(T, 5,7)
  blob_A = Mocha.cudnn_make_tensor_blob(T, 5,6, 1, 1)
  blob_B = Mocha.cudnn_make_tensor_blob(T, 6,7, 1, 1)
  blob_C = Mocha.cudnn_make_tensor_blob(T, 5,7, 1, 1)
  copy!(blob_A, A); copy!(blob_B, B); copy!(blob_C, C)
  CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, 5, 7, 6,
      convert(T,1.0), blob_A.ptr, 5, blob_B.ptr, 6, convert(T,1.0), blob_C.ptr, 5)
  result = zeros(T, 5,7)
  copy!(result, blob_C)
  @test all(-eps .< A*B+C - result .< eps)

  println("    > blas copy")
  x = rand(T, size(x))
  copy!(x_blob, x)
  CuBLAS.copy(sys.backend.cublas_ctx, T, length(x), x_blob.ptr.p, 1, x2_blob.ptr.p, 1)
  copy!(x2, x2_blob)
  @test all(-eps .< x2-x .< eps)
end

function test_cublas(sys::System)
  test_cublas(sys, Float32)
  test_cublas(sys, Float64)
end

test_cublas(sys_cudnn)
