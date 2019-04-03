# This file is a part of Julia. License is MIT: https://julialang.org/license

# extensions to Core types to add features in Base

# hook up VecElement constructor to Base.convert
function VecElement{T}(arg) where T
    VecElement{T}(convert(T, arg))
end
function convert(::Type{T}, arg) where T <: VecElement
    T(arg)
end
function convert(::Type{T}, arg::T) where T <: VecElement
    arg
end

# ## dims-type-converting Array constructors for convenience
# type and dimensionality specified, accepting dims as series of Integers
function Vector{T}(::UndefInitializer, m::Integer) where T
    Vector{T}(undef, Int(m))
end
function Matrix{T}(::UndefInitializer, m::Integer, n::Integer) where T
    Matrix{T}(undef, Int(m), Int(n))
end
function Array{T, N}(::UndefInitializer, d::Vararg{Integer, N}) where {T, N}
    Array{T, N}(undef, convert(Tuple{Vararg{Int}}, d))
end
# type but not dimensionality specified, accepting dims as series of Integers
function Array{T}(::UndefInitializer, m::Integer) where T
    Array{T, 1}(undef, Int(m))
end
function Array{T}(::UndefInitializer, m::Integer, n::Integer) where T
    Array{T, 2}(undef, Int(m), Int(n))
end
function Array{T}(::UndefInitializer, m::Integer, n::Integer, o::Integer) where T
    Array{T, 3}(undef, Int(m), Int(n), Int(o))
end
function Array{T}(::UndefInitializer, d::Integer...) where T
    Array{T}(undef, convert(Tuple{Vararg{Int}}, d))
end
# dimensionality but not type specified, accepting dims as series of Integers
function Vector(::UndefInitializer, m::Integer)
    Vector{Any}(undef, Int(m))
end
function Matrix(::UndefInitializer, m::Integer, n::Integer)
    Matrix{Any}(undef, Int(m), Int(n))
end
# Dimensions as a single tuple
function Array{T}(::UndefInitializer, d::NTuple{N, Integer}) where {T, N}
    Array{T, N}(undef, convert(Tuple{Vararg{Int}}, d))
end
function Array{T, N}(::UndefInitializer, d::NTuple{N, Integer}) where {T, N}
    Array{T, N}(undef, convert(Tuple{Vararg{Int}}, d))
end
# empty vector constructor
function Vector()
    Vector{Any}(undef, 0)
end

# Array constructors for nothing and missing
# type and dimensionality specified
function Array{T, N}(::Nothing, d...) where {T, N}
    fill!(Array{T, N}(undef, d...), nothing)
end
function Array{T, N}(::Missing, d...) where {T, N}
    fill!(Array{T, N}(undef, d...), missing)
end
# type but not dimensionality specified
function Array{T}(::Nothing, d...) where T
    fill!(Array{T}(undef, d...), nothing)
end
function Array{T}(::Missing, d...) where T
    fill!(Array{T}(undef, d...), missing)
end
