# A simple reference counting util

# import this function in Base so that we do not have conflicting definition
import Base.dec

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
