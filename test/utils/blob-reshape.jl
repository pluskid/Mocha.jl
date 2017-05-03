function test_blob_reshape(backend::Backend, T, eps)
  println("-- Testing blob reshape on $(typeof(backend)){$T}...")

  data = rand(T, 2,4,5,6)
  blob = make_blob(backend, data)
  blob2 = reshape_blob(backend, blob, length(data),1,1,1)
  data2 = rand(T, size(blob2))
  copy!(blob2, data2)
  copy!(data, blob)
  @test all(abs.(data - reshape(data2,size(data))) .< eps)
end

function test_blob_reshape(backend::Backend)
  test_blob_reshape(backend, Float32, 1e-10)
  test_blob_reshape(backend, Float64, 1e-10)
end

if test_cpu
  test_blob_reshape(backend_cpu)
end
if test_gpu
  test_blob_reshape(backend_gpu)
end
