using Benchmark

################################################################################
# A benchmark to show how broadcast_getindex performs with respect to plain
# loop. It turns out to be a bit faster.
#
# | Row | Function                | Average     | Relative | Replications |
# |-----|-------------------------|-------------|----------|--------------|
# | 1   | "mc_logistic_iter"      | 0.000258428 | 1.64526  | 30           |
# | 2   | "mc_logistic_broadcast" | 0.000157074 | 1.0      | 30           |
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


width, height, channels, num = (5, 6, 30, 128)

prob = abs(rand(width, height, channels, num))
label = abs(rand(Int, (width, height, 1, num))) % channels
label = convert(Array{Float64}, label)

mc_logistic_iter() = mc_logistic_iter(prob, label)
mc_logistic_broadcast() = mc_logistic_broadcast(prob, label)

@assert abs(mc_logistic_iter() - mc_logistic_broadcast()) < 1e-10

df = compare([mc_logistic_iter, mc_logistic_broadcast], 30)
println("$df")
