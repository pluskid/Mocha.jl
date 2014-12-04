function test_shared_blob(sys::System, T, eps)
  println("-- Testing shared blob on $(typeof(sys.backend)){$T}...")

  data = rand(T, 2,4,5,6)
  blob = make_blob(sys.backend, data)
  blob2 = reshape_blob(sys.backend, blob, length(data),1,1,1)
  data2 = rand(T, size(blob2))
  copy!(blob2, data2)
  copy!(data, blob)
  @test all(abs(data - reshape(data2,size(data))) .< eps)
end

function test_shared_blob(sys::System)
  test_shared_blob(sys, Float32, 1e-10)
  test_shared_blob(sys, Float64, 1e-10)
end

if test_cpu
  test_shared_blob(sys_cpu)
end
if test_cudnn
  test_shared_blob(sys_cudnn)
end
