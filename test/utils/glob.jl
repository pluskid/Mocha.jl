function test_glob()
  println("-- Testing glob Utilities")

  files = Mocha.glob("layers", r".*\.jl$")
  for file in files
    @test endswith(file, ".jl")
  end

  @test_throws ErrorException Mocha.glob("layers", r".*.jl$", sort_by=:foobar)

  files = Mocha.glob("layers", r".*\.jl$", sort_by=:name)
  for i = 1:length(files)-1
    @test files[i] <= files[i+1]
  end

  files = Mocha.glob("layers", r".*\.jl$", sort_by=:mtime)
  for i = 1:length(files)-1
    @test stat(joinpath("layers", files[i])).mtime <= stat(joinpath("layers", files[i+1])).mtime
  end
end

test_glob()
