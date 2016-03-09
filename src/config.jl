#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
export Config
module Config

println("Configuring Mocha...")
const use_cuda_key             = "MOCHA_USE_CUDA"
const cuda_dev_count_key       = "MOCHA_CUDA_DEVICE_COUNT"

const use_native_extension_key = "MOCHA_USE_NATIVE_EXT"

parseEnvInt(key; dflt=0) = begin
  try
    parse(Int, ENV[key])
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
const cuda_dev_count = parseEnvInt(cuda_dev_count_key, dflt=1)

const use_native_extension = parseEnvBool(use_native_extension_key)

use_cuda && println(" * CUDA       enabled [DEV_COUNT=$cuda_dev_count] ($use_cuda_key environment variable detected)")
use_cuda || println(" * CUDA       disabled by default")

use_native_extension && println(" * Native Ext enabled ($use_native_extension_key environment variable detected)")
use_native_extension || println(" * Native Ext disabled by default")

println("Mocha configured, continue loading module...")

end
