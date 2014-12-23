type MockResource
  resource :: Vector{Int}
  rc       :: RefCounter
end

function alloc_resource(n::Int)
  MockResource(zeros(Int, n), RefCounter(1))
end
function share_resource(r :: MockResource)
  MockResource(r.resource, ref(r.rc))
end
function free_resource(r :: MockResource)
  if dec(r.rc) == 0
    fill!(r.resource, -1)
  end
end

function test_ref_count()
  println("-- Testing simple reference counting...")

  res = alloc_resource(1)
  res2 = share_resource(res)

  free_resource(res)
  @test res2.resource[1] != -1

  free_resource(res2)
  @test res2.resource[1] == -1
end

test_ref_count()
