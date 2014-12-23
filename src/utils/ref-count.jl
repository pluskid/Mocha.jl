# A simple reference counting util
export RefCounter, inc, dec, ref

type RefCounter
  count :: Int
end

RefCounter() = RefCounter(0)

function inc(r :: RefCounter)
  r.count += 1
end
function dec(r :: RefCounter)
  r.count -= 1
  return r.count
end

function ref(r :: RefCounter)
  inc(r)
  r
end
