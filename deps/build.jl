sources = ["im2col.cpp", "pooling.cpp"]

compiler = get(ENV, "CXX", "g++")
flags    = ["-fPIC", "-Wall", "-O3", "-shared"]
libname  = "libmochaext.so"
openmp   = "-fopenmp"

@static is_apple() ? begin
  if !haskey(ENV, "MOCHA_FORCE_OMP")
    println("OpenMP is currently not officially supported by OS X Clang compiler yet.")
    println("(see http://clang-omp.github.io/ to install OpenMP clang extension, or")
    println("install gcc.")
    println("")
    println("Disabling OpenMP. To force enable OpenMP, set MOCHA_FORCE_OMP environment")
    println("variable and set the CXX environment variable to the appropriate value")
    println("to invoke GCC's g++ frontend, such as g++-5")
    println("")
    openmp = ""
  end
end : nothing

cmd = `$compiler $flags $openmp -o $libname $sources`
println("Running $cmd")
run(cmd)
