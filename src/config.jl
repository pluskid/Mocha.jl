export Config
module Config

println("Configuring Mocha...")
const use_cuda_key             = "MOCHA_USE_CUDA"
const cuda_dev_key             = "MOCHA_CUDA_DEVICE"

const use_opencl_key           = "MOCHA_USE_OPENCL"
const opencl_platform_key      = "MOCHA_OPENCL_PLATFORM"
const opencl_dev_key           = "MOCHA_OPENCL_DEVICE"

const use_native_extension_key = "MOCHA_USE_NATIVE_EXT"

parseEnvInt(key; dflt=0) = begin
  try
    parseint(ENV[key])
  catch
    dflt
  end
end

parseEnvBool(key; dflt=false) = begin
  try
    lowercase(ENV[key]) != "false"
  catch
    dflt
  end
end

const use_cuda = parseEnvBool(use_cuda_key)
const cuda_dev_id = parseEnvInt(cuda_dev_key)

const use_opencl = parseEnvBool(use_opencl_key)
const opencl_platform_id = parseEnvInt(opencl_platform_key)
const opencl_dev_id = parseEnvInt(opencl_dev_key)

const use_native_extension = parseEnvBool(use_native_extension_key)

use_cuda && println(" * CUDA       enabled [DEV=$cuda_dev_id] ($use_cuda_key environment variable detected)")
use_cuda || println(" * CUDA       disabled by default")

use_opencl && println(" * OpenCL     enabled [DEV=$opencl_platform_id:$opencl_dev_id] ($use_opencl_key environment variable detected)")
use_opencl || println(" * OpenCL     disabled by default")

use_native_extension && println(" * Native Ext enabled ($use_native_extension_key environment variable detected)")
use_native_extension || println(" * Native Ext disabled by default")

println("Mocha configured, continue loading module...")

end
