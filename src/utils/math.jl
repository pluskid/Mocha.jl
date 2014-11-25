export Vec
module Vec

# X[i] += a
function add_scal!{T}(X::Array{T}, a)
  leng = length(X)
  a = convert(eltype(X), a)
  @simd for i = 1:leng
    @inbounds X[i] += a
  end
end

# X[i] *= a
function mul_scal!{T}(X::Array{T}, a)
  leng = length(X)
  a = convert(eltype(X), a)
  @simd for i = 1:leng
    @inbounds X[i] *= a
  end
end

# X[i] *= Y[i]
function mul!{T}(X::Array{T}, Y::Array{T})
  leng = length(X)
  @simd for i = 1:leng
    @inbounds X[i] *= Y[i]
  end
end

# X[i] = X[i] / Y[i]
function div!{T}(X::Array{T}, Y::Array{T})
  leng = length(X)
  @simd for i = 1:leng
    @inbounds X[i] /= Y[i]
  end
end
# Y[i] = X[i] / Y[i]
function div2!{T}(X::Array{T}, Y::Array{T})
  leng = length(X)
  @simd for i = 1:leng
    @inbounds Y[i] = X[i] / Y[i]
  end
end

# X[i] = X[i]^p
function pow!{T}(X::Array{T}, p::Number)
  leng = length(X)
  @simd for i = 1:leng
    @inbounds X[i] = X[i]^p
  end
end

end # module Vec
