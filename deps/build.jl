sources = ["im2col.cpp", "pooling.cpp"]

compiler = "g++"
flags    = ["-fPIC", "-Wall", "-O3", "-shared"]
libname  = "libmochaext.so"
openmp   = "-fopenmp"

@osx? begin
  if !haskey(ENV, "MOCHA_FORCE_OMP")
    println("OpenMP is currently not officially supported by OS X Clang compiler yet.")
    println("(see http://clang-omp.github.io/ to install OpenMP clang extension, or")
    println("install gcc.")
    println("")
    println("Disabling OpenMP. To force enable OpenMP, set MOCHA_FORCE_OMP environment")
    println("variable.")
    println("")
    openmp = ""
  end
end : nothing

cmd = `$compiler $flags $openmp -o $libname $sources`
println("Running $cmd")
run(cmd)
