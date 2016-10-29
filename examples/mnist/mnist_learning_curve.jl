#!/usr/local/bin/julia

using PyPlot, JLD
# Plot the learning curve for the MNIST tutorial to verify
# the solver is progressing toward convergence

stats = load("snapshots/statistics.jld")
# println(typeof(stats))

tables = stats["statistics"]
ov = tables["obj_val"]
xy = sort(collect(ov))
datapoints = size(xy[:])
println("XY contains $datapoints datapoints")
x = [i for (i,j) in xy]
y = [j for (i,j) in xy]
x = convert(Array{Int64}, x)
y = convert(Array{Float64}, y)

raw = plot(x, y, linewidth=1, label="Raw")
xlabel("Iterations")
ylabel("Objective Value")
title("MNIST Learning Curve")
grid("on")

function low_pass{T <: Real}(x::Vector{T}, window::Int)
  len = length(x)
  y = Vector{Float64}(len)
  for i in 1:len
      # I want the mean of the first i terms up to width of window
      # Putting some numbers to this with window 4 
      # i win lo  hi
      # 1  4  1   1  
      # 2  4  1   2 
      # 3  4  1   3 
      # 4  4  1   4
      # 5  4  1   5
      # 6  4  2   6  => window starts to slide
      lo = max(1, i - window)
      hi = i
      y[i] = mean(x[lo:hi])
  end
  return y
end

window = Int64(round(length(xy)/4.0))
y_avg = low_pass(y, window)
avg = plot(x, y_avg, linewidth=2, label="Low Pass")
legend(handles=[raw; avg])
show()  #Shows the final plot
