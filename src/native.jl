export Native
module Native
using ..Mocha
using Compat

const libname = "libmochaext.so"
const libpath = abspath(joinpath(dirname(@__FILE__), "..", "deps", libname))
try
  println("Loading native extension $libname...")
  const global library = Libdl.dlopen(libpath)
catch y
  println("ERROR: Could not load native extension at $libpath.")
  println("To use native extension, run deps/build.jl to compile the native code.")
  throw(y)
end
println("Native extension loaded")

end # module Native
