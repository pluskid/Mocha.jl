using Benchmark

################################################################################
# A benchmark to show how broadcast_getindex performs with respect to plain
# loop. It turns out to be a bit faster. @inbounds annotation helps a little
# bit here.
#
# | Row | Function                    | Average     | Relative | Replications |
# |-----|-----------------------------|-------------|----------|--------------|
# | 1   | "mc_logistic_iter"          | 0.000492798 | 3.62449  | 100          |
# | 2   | "mc_logistic_iter_inbounds" | 0.000239364 | 1.7605   | 100          |
# | 3   | "mc_logistic_broadcast"     | 0.000135963 | 1.0      | 100          |
################################################################################

function mc_logistic_broadcast(pred, label)
  width, height, channels, num = size(pred)
  loss = sum(-log(broadcast_getindex(pred,
      reshape(1:width, (width,1,1,1)),
      reshape(1:height, (1,height,1,1)),
      int(label) + 1,
      reshape(1:num, (1,1,1,num)))))
  return loss / (width*height*num)
end

function mc_logistic_iter(pred, label)
  width, height, channels, num = size(pred)
  loss = 0
  for w = 1:width
    for h = 1:height
      for n = 1:num
        idx = int(label[w,h,1,n])+1
        loss += -log(pred[w,h,idx,n])
      end
    end
  end
  return loss / (width*height*num)
end

function mc_logistic_iter_inbounds(pred, label)
  width, height, channels, num = size(pred)
  loss = 0
  for w = 1:width
    for h = 1:height
      @simd for n = 1:num
        @inbounds loss += -log(pred[w,h,int(label[w,h,1,n])+1,n])
      end
    end
  end
  return loss / (width*height*num)
end

width, height, channels, num = (5, 6, 30, 128)

prob = abs(rand(width, height, channels, num))
label = abs(rand(Int, (width, height, 1, num))) % channels
label = convert(Array{Float64}, label)

mc_logistic_iter() = mc_logistic_iter(prob, label)
mc_logistic_broadcast() = mc_logistic_broadcast(prob, label)
mc_logistic_iter_inbounds() = mc_logistic_iter_inbounds(prob, label)

@assert abs(mc_logistic_iter() - mc_logistic_broadcast()) < 1e-10
@assert abs(mc_logistic_iter_inbounds() - mc_logistic_broadcast()) < 1e-10

df = compare([mc_logistic_iter, mc_logistic_iter_inbounds, mc_logistic_broadcast], 100)
println("$df")
