# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    Rational{T<:Integer} <: Real

Rational number type, with numerator and denominator of type `T`.
"""
struct Rational{T<:Integer} <: Real
    num::T
    den::T

    function Rational{T}(num::Integer, den::Integer) where T<:Integer
        num == den == zero(T) && __throw_rational_argerror(T)
        num2, den2 = (sign(den) < 0) ? divgcd(-num, -den) : divgcd(num, den)
        new(num2, den2)
    end
end
@noinline __throw_rational_argerror(T) = throw(ArgumentError("invalid rational: zero($T)//zero($T)"))

function Rational(n::T, d::T) where T <: Integer
    Rational{T}(n, d)
end
function Rational(n::Integer, d::Integer)
    Rational(promote(n, d)...)
end
function Rational(n::Integer)
    Rational(n, one(n))
end

function divgcd(x::Integer,y::Integer)
    g = gcd(x,y)
    div(x,g), div(y,g)
end

"""
    //(num, den)

Divide two integers or rational numbers, giving a [`Rational`](@ref) result.

# Examples
```jldoctest
julia> 3 // 5
3//5

julia> (3 // 5) // (2 // 1)
3//10
```
"""
//(n::Integer,  d::Integer) = Rational(n,d)

function //(x::Rational, y::Integer)
    xn,yn = divgcd(x.num,y)
    xn//checked_mul(x.den,yn)
end
function //(x::Integer,  y::Rational)
    xn,yn = divgcd(x,y.num)
    checked_mul(xn,y.den)//yn
end
function //(x::Rational, y::Rational)
    xn,yn = divgcd(x.num,y.num)
    xd,yd = divgcd(x.den,y.den)
    checked_mul(xn,yd)//checked_mul(xd,yn)
end

function //(x::Complex, y::Real)
    complex(real(x) // y, imag(x) // y)
end
function //(x::Number, y::Complex)
    x * conj(y) // abs2(y)
end


function //(X::AbstractArray, y::Number)
    X .// y
end

function show(io::IO, x::Rational)
    show(io, numerator(x))
    print(io, "//")
    show(io, denominator(x))
end

function read(s::IO, ::Type{Rational{T}}) where T<:Integer
    r = read(s,T)
    i = read(s,T)
    r//i
end
function write(s::IO, z::Rational)
    write(s,numerator(z),denominator(z))
end

function Rational{T}(x::Rational) where T <: Integer
    Rational{T}(convert(T, x.num), convert(T, x.den))
end
function Rational{T}(x::Integer) where T <: Integer
    Rational{T}(convert(T, x), convert(T, 1))
end

function Rational(x::Rational)
    x
end

function Bool(x::Rational)
    if x == 0
        false
    else
        if x == 1
            true
        else
            throw(InexactError(:Bool, Bool, x))
        end
    end
end # to resolve ambiguity
function (::Type{T})(x::Rational) where T <: Integer
    if isinteger(x)
        convert(T, x.num)
    else
        throw(InexactError(nameof(T), T, x))
    end
end

function AbstractFloat(x::Rational)
    float(x.num) / float(x.den)
end
function (::Type{T})(x::Rational{S}) where T<:AbstractFloat where S
    P = promote_type(T,S)
    convert(T, convert(P,x.num)/convert(P,x.den))
end

function Rational{T}(x::AbstractFloat) where T<:Integer
    r = rationalize(T, x, tol=0)
    x == convert(typeof(x), r) || throw(InexactError(:Rational, Rational{T}, x))
    r
end
function Rational(x::Float64)
    Rational{Int64}(x)
end
function Rational(x::Float32)
    Rational{Int}(x)
end

function big(q::Rational)
    big(numerator(q)) // big(denominator(q))
end

function big(z::Complex{<:Rational{<:Integer}})
    Complex{Rational{BigInt}}(z)
end

function promote_rule(::Type{Rational{T}}, ::Type{S}) where {T <: Integer, S <: Integer}
    Rational{promote_type(T, S)}
end
function promote_rule(::Type{Rational{T}}, ::Type{Rational{S}}) where {T <: Integer, S <: Integer}
    Rational{promote_type(T, S)}
end
function promote_rule(::Type{Rational{T}}, ::Type{S}) where {T <: Integer, S <: AbstractFloat}
    promote_type(T, S)
end

function widen(::Type{Rational{T}}) where T
    Rational{widen(T)}
end

"""
    rationalize([T<:Integer=Int,] x; tol::Real=eps(x))

Approximate floating point number `x` as a [`Rational`](@ref) number with components
of the given integer type. The result will differ from `x` by no more than `tol`.

# Examples
```jldoctest
julia> rationalize(5.6)
28//5

julia> a = rationalize(BigInt, 10.3)
103//10

julia> typeof(numerator(a))
BigInt
```
"""
function rationalize(::Type{T}, x::AbstractFloat, tol::Real) where T<:Integer
    if tol < 0
        throw(ArgumentError("negative tolerance $tol"))
    end
    isnan(x) && return T(x)//one(T)
    isinf(x) && return (x < 0 ? -one(T) : one(T))//zero(T)

    p,  q  = (x < 0 ? -one(T) : one(T)), zero(T)
    pp, qq = zero(T), one(T)

    x = abs(x)
    a = trunc(x)
    r = x-a
    y = one(x)

    tolx = oftype(x, tol)
    nt, t, tt = tolx, zero(tolx), tolx
    ia = np = nq = zero(T)

    # compute the successive convergents of the continued fraction
    #  np // nq = (p*a + pp) // (q*a + qq)
    while r > nt
        try
            ia = convert(T,a)

            np = checked_add(checked_mul(ia,p),pp)
            nq = checked_add(checked_mul(ia,q),qq)
            p, pp = np, p
            q, qq = nq, q
        catch e
            isa(e,InexactError) || isa(e,OverflowError) || rethrow()
            return p // q
        end

        # naive approach of using
        #   x = 1/r; a = trunc(x); r = x - a
        # is inexact, so we store x as x/y
        x, y = y, r
        a, r = divrem(x,y)

        # maintain
        # x0 = (p + (-1)^i * r) / q
        t, tt = nt, t
        nt = a*t+tt
    end

    # find optimal semiconvergent
    # smallest a such that x-a*y < a*t+tt
    a = cld(x-tt,y+t)
    try
        ia = convert(T,a)
        np = checked_add(checked_mul(ia,p),pp)
        nq = checked_add(checked_mul(ia,q),qq)
        return np // nq
    catch e
        isa(e,InexactError) || isa(e,OverflowError) || rethrow()
        return p // q
    end
end
function rationalize(::Type{T}, x::AbstractFloat; tol::Real=eps(x)) where T <: Integer
    rationalize(T, x, tol)::Rational{T}
end
function rationalize(x::AbstractFloat; kvs...)
    rationalize(Int, x; kvs...)
end

"""
    numerator(x)

Numerator of the rational representation of `x`.

# Examples
```jldoctest
julia> numerator(2//3)
2

julia> numerator(4)
4
```
"""
numerator(x::Integer) = x
function numerator(x::Rational)
    x.num
end

"""
    denominator(x)

Denominator of the rational representation of `x`.

# Examples
```jldoctest
julia> denominator(2//3)
3

julia> denominator(4)
1
```
"""
denominator(x::Integer) = one(x)
function denominator(x::Rational)
    x.den
end

function sign(x::Rational)
    oftype(x, sign(x.num))
end
function signbit(x::Rational)
    signbit(x.num)
end
function copysign(x::Rational, y::Real)
    copysign(x.num, y) // x.den
end
function copysign(x::Rational, y::Rational)
    copysign(x.num, y.num) // x.den
end

function abs(x::Rational)
    Rational(abs(x.num), x.den)
end

function typemin(::Type{Rational{T}}) where T <: Integer
    -(one(T)) // zero(T)
end
function typemax(::Type{Rational{T}}) where T <: Integer
    one(T) // zero(T)
end

function isinteger(x::Rational)
    x.den == 1
end

function -(x::Rational)
    -(x.num) // x.den
end
function -(x::Rational{T}) where T<:BitSigned
    x.num == typemin(T) && throw(OverflowError("rational numerator is typemin(T)"))
    (-x.num) // x.den
end
function -(x::Rational{T}) where T<:Unsigned
    x.num != zero(T) && throw(OverflowError("cannot negate unsigned number"))
    x
end

for (op,chop) in ((:+,:checked_add), (:-,:checked_sub),
                  (:rem,:rem), (:mod,:mod))
    @eval begin
        function ($op)(x::Rational, y::Rational)
            xd, yd = divgcd(x.den, y.den)
            Rational(($chop)(checked_mul(x.num,yd), checked_mul(y.num,xd)), checked_mul(x.den,yd))
        end
    end
end

function *(x::Rational, y::Rational)
    xn,yd = divgcd(x.num,y.den)
    xd,yn = divgcd(x.den,y.num)
    checked_mul(xn,yn) // checked_mul(xd,yd)
end
function /(x::Rational, y::Rational)
    x // y
end
function /(x::Rational, y::Complex{<:Union{Integer, Rational}})
    x // y
end
function inv(x::Rational)
    Rational(x.den, x.num)
end

function fma(x::Rational, y::Rational, z::Rational)
    x * y + z
end

function ==(x::Rational, y::Rational)
    (x.den == y.den) & (x.num == y.num)
end
function <(x::Rational, y::Rational)
    if x.den == y.den
        x.num < y.num
    else
        widemul(x.num, y.den) < widemul(x.den, y.num)
    end
end
function <=(x::Rational, y::Rational)
    if x.den == y.den
        x.num <= y.num
    else
        widemul(x.num, y.den) <= widemul(x.den, y.num)
    end
end


function ==(x::Rational, y::Integer)
    (x.den == 1) & (x.num == y)
end
function ==(x::Integer, y::Rational)
    y == x
end
function <(x::Rational, y::Integer)
    x.num < widemul(x.den, y)
end
function <(x::Integer, y::Rational)
    widemul(x, y.den) < y.num
end
function <=(x::Rational, y::Integer)
    x.num <= widemul(x.den, y)
end
function <=(x::Integer, y::Rational)
    widemul(x, y.den) <= y.num
end

function ==(x::AbstractFloat, q::Rational)
    if isfinite(x)
        (count_ones(q.den) == 1) & (x*q.den == q.num)
    else
        x == q.num/q.den
    end
end

function ==(q::Rational, x::AbstractFloat)
    x == q
end

for rel in (:<,:<=,:cmp)
    for (Tx,Ty) in ((Rational,AbstractFloat), (AbstractFloat,Rational))
        @eval function ($rel)(x::$Tx, y::$Ty)
            if isnan(x)
                $(rel == :cmp ? :(return isnan(y) ? 0 : 1) :
                                :(return false))
            end
            if isnan(y)
                $(rel == :cmp ? :(return -1) :
                                :(return false))
            end

            xn, xp, xd = decompose(x)
            yn, yp, yd = decompose(y)

            if xd < 0
                xn = -xn
                xd = -xd
            end
            if yd < 0
                yn = -yn
                yd = -yd
            end

            xc, yc = widemul(xn,yd), widemul(yn,xd)
            xs, ys = sign(xc), sign(yc)

            if xs != ys
                return ($rel)(xs,ys)
            elseif xs == 0
                # both are zero or ±Inf
                return ($rel)(xn,yn)
            end

            xb, yb = ndigits0z(xc,2) + xp, ndigits0z(yc,2) + yp

            if xb == yb
                xc, yc = promote(xc,yc)
                if xp > yp
                    xc = (xc<<(xp-yp))
                else
                    yc = (yc<<(yp-xp))
                end
                return ($rel)(xc,yc)
            else
                return xc > 0 ? ($rel)(xb,yb) : ($rel)(yb,xb)
            end
        end
    end
end

# needed to avoid ambiguity between ==(x::Real, z::Complex) and ==(x::Rational, y::Number)
function ==(z::Complex, x::Rational)
    isreal(z) & (real(z) == x)
end
function ==(x::Rational, z::Complex)
    isreal(z) & (real(z) == x)
end

for op in (:div, :fld, :cld)
    @eval begin
        function ($op)(x::Rational, y::Integer )
            xn,yn = divgcd(x.num,y)
            ($op)(xn, checked_mul(x.den,yn))
        end
        function ($op)(x::Integer,  y::Rational)
            xn,yn = divgcd(x,y.num)
            ($op)(checked_mul(xn,y.den), yn)
        end
        function ($op)(x::Rational, y::Rational)
            xn,yn = divgcd(x.num,y.num)
            xd,yd = divgcd(x.den,y.den)
            ($op)(checked_mul(xn,yd), checked_mul(xd,yn))
        end
    end
end

function trunc(::Type{T}, x::Rational) where T
    convert(T, div(x.num, x.den))
end
function floor(::Type{T}, x::Rational) where T
    convert(T, fld(x.num, x.den))
end
function ceil(::Type{T}, x::Rational) where T
    convert(T, cld(x.num, x.den))
end
function round(::Type{T}, x::Rational, r::RoundingMode=RoundNearest) where T
    _round_rational(T, x, r)
end
function round(x::Rational, r::RoundingMode)
    round(Rational, x, r)
end

function _round_rational(::Type{T}, x::Rational{Tr}, ::RoundingMode{:Nearest}) where {T,Tr}
    if denominator(x) == zero(Tr) && T <: Integer
        throw(DivideError())
    elseif denominator(x) == zero(Tr)
        return convert(T, copysign(one(Tr)//zero(Tr), numerator(x)))
    end
    q,r = divrem(numerator(x), denominator(x))
    s = q
    if abs(r) >= abs((denominator(x)-copysign(Tr(4), numerator(x))+one(Tr)+iseven(q))>>1 + copysign(Tr(2), numerator(x)))
        s += copysign(one(Tr),numerator(x))
    end
    convert(T, s)
end

function _round_rational(::Type{T}, x::Rational{Tr}, ::RoundingMode{:NearestTiesAway}) where {T,Tr}
    if denominator(x) == zero(Tr) && T <: Integer
        throw(DivideError())
    elseif denominator(x) == zero(Tr)
        return convert(T, copysign(one(Tr)//zero(Tr), numerator(x)))
    end
    q,r = divrem(numerator(x), denominator(x))
    s = q
    if abs(r) >= abs((denominator(x)-copysign(Tr(4), numerator(x))+one(Tr))>>1 + copysign(Tr(2), numerator(x)))
        s += copysign(one(Tr),numerator(x))
    end
    convert(T, s)
end

function _round_rational(::Type{T}, x::Rational{Tr}, ::RoundingMode{:NearestTiesUp}) where {T,Tr}
    if denominator(x) == zero(Tr) && T <: Integer
        throw(DivideError())
    elseif denominator(x) == zero(Tr)
        return convert(T, copysign(one(Tr)//zero(Tr), numerator(x)))
    end
    q,r = divrem(numerator(x), denominator(x))
    s = q
    if abs(r) >= abs((denominator(x)-copysign(Tr(4), numerator(x))+one(Tr)+(numerator(x)<0))>>1 + copysign(Tr(2), numerator(x)))
        s += copysign(one(Tr),numerator(x))
    end
    convert(T, s)
end

function round(::Type{T}, x::Rational{Bool}, ::RoundingMode=RoundNearest) where T
    if denominator(x) == false && (T <: Union{Integer, Bool})
        throw(DivideError())
    end
    convert(T, x)
end

function trunc(x::Rational{T}) where T
    Rational(trunc(T, x))
end
function floor(x::Rational{T}) where T
    Rational(floor(T, x))
end
function ceil(x::Rational{T}) where T
    Rational(ceil(T, x))
end
function round(x::Rational{T}) where T
    Rational(round(T, x))
end

function ^(x::Rational, n::Integer)
    n >= 0 ? power_by_squaring(x,n) : power_by_squaring(inv(x),-n)
end

function ^(x::Number, y::Rational)
    x ^ (y.num / y.den)
end
function (x::T ^ y::Rational) where T <: AbstractFloat
    x ^ convert(T, y)
end
function (z::Complex{T} ^ p::Rational) where T <: Real
    z ^ convert(typeof(one(T) ^ p), p)
end

function ^(z::Complex{<:Rational}, n::Bool)
    if n
        z
    else
        one(z)
    end
end # to resolve ambiguity
function ^(z::Complex{<:Rational}, n::Integer)
    n >= 0 ? power_by_squaring(z,n) : power_by_squaring(inv(z),-n)
end

function iszero(x::Rational)
    iszero(numerator(x))
end
function isone(x::Rational)
    isone(numerator(x)) & isone(denominator(x))
end

function lerpi(j::Integer, d::Integer, a::Rational, b::Rational)
    ((d-j)*a)/d + (j*b)/d
end

function float(::Type{Rational{T}}) where T <: Integer
    float(T)
end
