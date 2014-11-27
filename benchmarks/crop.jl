using Benchmark
################################################################################
# Benchmark to see a better way of implementing crop
#
# | Row | Function     | Average   | Relative | Replications |
# |-----|--------------|-----------|----------|--------------|
# | 1   | "crop_loop"  | 0.0260486 | 1.0      | 50           |
# | 2   | "crop_whole" | 0.109872  | 4.21796  | 50           |
################################################################################

function crop_loop(input, crop_wh, output)
  w_off = div(size(input,1) - crop_wh[1], 2)
  h_off = div(size(input,2) - crop_wh[2], 2)

  for n = 1:size(input,4)
    for c = 1:size(input,3)
      for h = 1:crop_wh[2]
        @simd for w = 1:crop_wh[1]
          @inbounds output[w,h,c,n] = input[w+w_off,h+h_off,c,n]
        end
      end
    end
  end
end

function crop_whole(input, crop_wh, output)
  w_off = div(size(input,1) - crop_wh[1], 2)
  h_off = div(size(input,2) - crop_wh[2], 2)

  output[:] = input[w_off+1:w_off+crop_wh[1], h_off+1:h_off+crop_wh[2],:,:]
end

input = rand(256, 256, 3, 128)
crop_wh = [227, 227]
out1 = zeros(crop_wh..., size(input,3), size(input,4))
out2 = similar(out1)
crop_loop(input, crop_wh, out1)
crop_whole(input, crop_wh, out2)
@assert all(abs(out1-out2) .< 1e-10)

crop_loop() = crop_loop(input, crop_wh, out1)
crop_whole() = crop_whole(input, crop_wh, out2)

df = compare([crop_loop, crop_whole], 50)
println("$df")

