export Config
module Config

println("Configuring Mocha...")
const use_cuda_key = "MOCHA_USE_CUDA"
const cuda_dev_key = "MOCHA_GPU_DEVICE"
const use_native_extension_key = "MOCHA_USE_NATIVE_EXT"

if haskey(ENV, cuda_dev_key)
  const cuda_dev_id =
  try
    parse(Int, ENV[cuda_dev_key])
  catch
    0
  end
else
  const cuda_dev_id = 0
end

if haskey(ENV, use_cuda_key) && lowercase(ENV[use_cuda_key]) != "false"
  println(" * CUDA       enabled [DEV=$cuda_dev_id] ($use_cuda_key environment variable detected)")
  const use_cuda = true
else
  println(" * CUDA       disabled by default")
  const use_cuda = false
end

if haskey(ENV, use_native_extension_key) && lowercase(ENV[use_native_extension_key]) != "false"
  println(" * Native Ext enabled ($use_native_extension_key environment variable detected)")
  const use_native_extension = true
else
  println(" * Native Ext disabled by default")
  const use_native_extension = false
end
println("Mocha configured, continue loading module...")

end
