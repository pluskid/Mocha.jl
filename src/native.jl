export Native
module Native

libpath = abspath(joinpath(dirname(@__FILE__), "..", "deps", "libmochaext.so"))
try
  library = dlopen(libpath)
  available = true
catch
  available = false
end


end # module Native
