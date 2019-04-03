# This file is a part of Julia. License is MIT: https://julialang.org/license

## generic operations on numbers ##

# Numbers are convertible
function convert(::Type{T}, x::T) where T <: Number
    x
end
function convert(::Type{T}, x::Number) where T <: Number
    T(x)
end

"""
    isinteger(x) -> Bool

Test whether `x` is numerically equal to some integer.

# Examples
```jldoctest
julia> isinteger(4.0)
true
```
"""
isinteger(x::Integer) = true

"""
    iszero(x)

Return `true` if `x == zero(x)`; if `x` is an array, this checks whether
all of the elements of `x` are zero.

# Examples
```jldoctest
julia> iszero(0.0)
true

julia> iszero([1, 9, 0])
false

julia> iszero([false, 0, 0])
true
```
"""
iszero(x) = x == zero(x) # fallback method

"""
    isone(x)

Return `true` if `x == one(x)`; if `x` is an array, this checks whether
`x` is an identity matrix.

# Examples
```jldoctest
julia> isone(1.0)
true

julia> isone([1 0; 0 2])
false

julia> isone([1 0; 0 true])
true
```
"""
isone(x) = x == one(x) # fallback method

function size(x::Number)
    ()
end
function size(x::Number, d::Integer)
    if d < 1
        throw(BoundsError())
    else
        1
    end
end
function axes(x::Number)
    ()
end
function axes(x::Number, d::Integer)
    if d < 1
        throw(BoundsError())
    else
        OneTo(1)
    end
end
function eltype(::Type{T}) where T <: Number
    T
end
function ndims(x::Number)
    0
end
function ndims(::Type{<:Number})
    0
end
function length(x::Number)
    1
end
function firstindex(x::Number)
    1
end
function lastindex(x::Number)
    1
end
function IteratorSize(::Type{<:Number})
    HasShape{0}()
end
function keys(::Number)
    OneTo(1)
end

function getindex(x::Number)
    x
end
function getindex(x::Number, i::Integer)
    @_inline_meta
    @boundscheck i == 1 || throw(BoundsError())
    x
end
function getindex(x::Number, I::Integer...)
    @_inline_meta
    @boundscheck all([i == 1 for i in I]) || throw(BoundsError())
    x
end
function first(x::Number)
    x
end
function last(x::Number)
    x
end
function copy(x::Number)
    x
end # some code treats numbers as collection-like

"""
    divrem(x, y)

The quotient and remainder from Euclidean division. Equivalent to `(div(x,y), rem(x,y))` or
`(x÷y, x%y)`.

# Examples
```jldoctest
julia> divrem(3,7)
(0, 3)

julia> divrem(7,3)
(2, 1)
```
"""
divrem(x,y) = (div(x,y),rem(x,y))

"""
    fldmod(x, y)

The floored quotient and modulus after division. Equivalent to `(fld(x,y), mod(x,y))`.
"""
fldmod(x,y) = (fld(x,y),mod(x,y))

"""
    signbit(x)

Returns `true` if the value of the sign of `x` is negative, otherwise `false`.

# Examples
```jldoctest
julia> signbit(-4)
true

julia> signbit(5)
false

julia> signbit(5.5)
false

julia> signbit(-4.1)
true
```
"""
signbit(x::Real) = x < 0

"""
    sign(x)

Return zero if `x==0` and ``x/|x|`` otherwise (i.e., ±1 for real `x`).
"""
sign(x::Number) = x == 0 ? x/abs(oneunit(x)) : x/abs(x)
function sign(x::Real)
    ifelse(x < 0, oftype(one(x), -1), ifelse(x > 0, one(x), (typeof(one(x)))(x)))
end
function sign(x::Unsigned)
    ifelse(x > 0, one(x), oftype(one(x), 0))
end
function abs(x::Real)
    ifelse(signbit(x), -x, x)
end

"""
    abs2(x)

Squared absolute value of `x`.

# Examples
```jldoctest
julia> abs2(-3)
9
```
"""
abs2(x::Real) = x*x

"""
    flipsign(x, y)

Return `x` with its sign flipped if `y` is negative. For example `abs(x) = flipsign(x,x)`.

# Examples
```jldoctest
julia> flipsign(5, 3)
5

julia> flipsign(5, -3)
-5
```
"""
flipsign(x::Real, y::Real) = ifelse(signbit(y), -x, +x) # the + is for type-stability on Bool

"""
    copysign(x, y) -> z

Return `z` which has the magnitude of `x` and the same sign as `y`.

# Examples
```jldoctest
julia> copysign(1, -2)
-1

julia> copysign(-1, 2)
1
```
"""
copysign(x::Real, y::Real) = ifelse(signbit(x)!=signbit(y), -x, +x)

function conj(x::Real)
    x
end
function transpose(x::Number)
    x
end
function adjoint(x::Number)
    conj(x)
end
function angle(z::Real)
    atan(zero(z), z)
end

"""
    inv(x)

Return the multiplicative inverse of `x`, such that `x*inv(x)` or `inv(x)*x`
yields [`one(x)`](@ref) (the multiplicative identity) up to roundoff errors.

If `x` is a number, this is essentially the same as `one(x)/x`, but for
some types `inv(x)` may be slightly more efficient.

# Examples
```jldoctest
julia> inv(2)
0.5

julia> inv(1 + 2im)
0.2 - 0.4im

julia> inv(1 + 2im) * (1 + 2im)
1.0 + 0.0im

julia> inv(2//3)
3//2
```

!!! compat "Julia 1.2"
    `inv(::Missing)` requires at least Julia 1.2.
"""
inv(x::Number) = one(x)/x


"""
    widemul(x, y)

Multiply `x` and `y`, giving the result as a larger type.

# Examples
```jldoctest
julia> widemul(Float32(3.), 4.)
12.0
```
"""
widemul(x::Number, y::Number) = widen(x)*widen(y)

function iterate(x::Number)
    (x, nothing)
end
function iterate(x::Number, ::Any)
    nothing
end
function isempty(x::Number)
    false
end
function in(x::Number, y::Number)
    x == y
end

function map(f, x::Number, ys::Number...)
    f(x, ys...)
end

"""
    zero(x)

Get the additive identity element for the type of `x` (`x` can also specify the type itself).

# Examples
```jldoctest
julia> zero(1)
0

julia> zero(big"2.0")
0.0

julia> zero(rand(2,2))
2×2 Array{Float64,2}:
 0.0  0.0
 0.0  0.0
```
"""
zero(x::Number) = oftype(x,0)
function zero(::Type{T}) where T <: Number
    convert(T, 0)
end

"""
    one(x)
    one(T::type)

Return a multiplicative identity for `x`: a value such that
`one(x)*x == x*one(x) == x`.  Alternatively `one(T)` can
take a type `T`, in which case `one` returns a multiplicative
identity for any `x` of type `T`.

If possible, `one(x)` returns a value of the same type as `x`,
and `one(T)` returns a value of type `T`.  However, this may
not be the case for types representing dimensionful quantities
(e.g. time in days), since the multiplicative
identity must be dimensionless.  In that case, `one(x)`
should return an identity value of the same precision
(and shape, for matrices) as `x`.

If you want a quantity that is of the same type as `x`, or of type `T`,
even if `x` is dimensionful, use [`oneunit`](@ref) instead.

# Examples
```jldoctest
julia> one(3.7)
1.0

julia> one(Int)
1

julia> import Dates; one(Dates.Day(1))
1
```
"""
one(::Type{T}) where {T<:Number} = convert(T,1)
function one(x::T) where T <: Number
    one(T)
end
# note that convert(T, 1) should throw an error if T is dimensionful,
# so this fallback definition should be okay.

"""
    oneunit(x::T)
    oneunit(T::Type)

Returns `T(one(x))`, where `T` is either the type of the argument or
(if a type is passed) the argument.  This differs from [`one`](@ref) for
dimensionful quantities: `one` is dimensionless (a multiplicative identity)
while `oneunit` is dimensionful (of the same type as `x`, or of type `T`).

# Examples
```jldoctest
julia> oneunit(3.7)
1.0

julia> import Dates; oneunit(Dates.Day)
1 day
```
"""
oneunit(x::T) where {T} = T(one(x))
function oneunit(::Type{T}) where T
    T(one(T))
end

"""
    big(T::Type)

Compute the type that represents the numeric type `T` with arbitrary precision.
Equivalent to `typeof(big(zero(T)))`.

# Examples
```jldoctest
julia> big(Rational)
Rational{BigInt}

julia> big(Float64)
BigFloat

julia> big(Complex{Int})
Complex{BigInt}
```
"""
big(::Type{T}) where {T<:Number} = typeof(big(zero(T)))
