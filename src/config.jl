export Config
module Config
import ..info

info("Configuring Mocha...")
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
  info(" * CUDA       enabled [DEV=$cuda_dev_id] ($use_cuda_key environment variable detected)")
  const use_cuda = true
else
  info(" * CUDA       disabled by default")
  const use_cuda = false
end

if haskey(ENV, use_native_extension_key) && lowercase(ENV[use_native_extension_key]) != "false"
  info(" * Native Ext enabled ($use_native_extension_key environment variable detected)")
  const use_native_extension = true
else
  info(" * Native Ext disabled by default")
  const use_native_extension = false
end
info("Mocha configured, continue loading module...")

end
