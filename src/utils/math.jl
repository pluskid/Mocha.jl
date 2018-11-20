export Vec
module Vec

# X[i] += a
function add_scal!(X::Array{T}, a) where {T}
  leng = length(X)
  a = convert(eltype(X), a)
  @simd for i = 1:leng
    @inbounds X[i] += a
  end
end

# X[i] *= a
function mul_scal!(X::Array{T}, a) where {T}
  leng = length(X)
  a = convert(eltype(X), a)
  @simd for i = 1:leng
    @inbounds X[i] *= a
  end
end

# X[i] *= Y[i]
function mul!(X::Array{T}, Y::Array{T}) where {T}
  leng = length(X)
  @simd for i = 1:leng
    @inbounds X[i] *= Y[i]
  end
end

# X[i] = X[i] / Y[i]
function div!(X::Array{T}, Y::Array{T}) where {T}
  leng = length(X)
  @simd for i = 1:leng
    @inbounds X[i] /= Y[i]
  end
end
# Y[i] = X[i] / Y[i]
function div2!(X::Array{T}, Y::Array{T}) where {T}
  leng = length(X)
  @simd for i = 1:leng
    @inbounds Y[i] = X[i] / Y[i]
  end
end

# X[i] = X[i]^p
function pow!(X::Array{T}, p::Number) where {T}
  leng = length(X)
  @simd for i = 1:leng
    @inbounds X[i] = X[i]^p
  end
end

end # module Vec
