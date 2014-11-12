export Native
module Native
import ..info
import ..critical

const libname = "libmochaext.so"
const libpath = abspath(joinpath(dirname(@__FILE__), "..", "deps", libname))
try
  info("Loading native extension $libname...")
  const global library = dlopen(libpath)
catch y
  critical("ERROR: Could not load native extension at $libpath.")
  critical("To use native extension, run deps/build.jl to compile the native code.")
  throw(y)
end
info("Native extension loaded")

end # module Native
