export Config
module Config
import ..info

info("Configuring Mocha...")
const use_cuda_key = "MOCHA_USE_CUDA"
const use_native_extension_key = "MOCHA_USE_NATIVE_EXT"

if haskey(ENV, use_cuda_key)
  info(" * CUDA             enabled ($use_cuda_key environment variable detected)")
  const use_cuda = true
else
  info(" * CUDA             disabled by default")
  const use_cuda = false
end

if haskey(ENV, use_native_extension_key)
  info(" * Native Extension enabled ($use_native_extension_key environment variable detected)")
  const use_native_extension = true
else
  info(" * Native Extension disabled by default")
  const use_native_extension = false
end
info("Mocha configured, continue loading module...")

end
