# This file is a part of Julia. License is MIT: https://julialang.org/license

const IEEEFloat = Union{Float16, Float32, Float64}

## floating point traits ##

"""
    Inf16

Positive infinity of type [`Float16`](@ref).
"""
const Inf16 = bitcast(Float16, 0x7c00)
"""
    NaN16

A not-a-number value of type [`Float16`](@ref).
"""
const NaN16 = bitcast(Float16, 0x7e00)
"""
    Inf32

Positive infinity of type [`Float32`](@ref).
"""
const Inf32 = bitcast(Float32, 0x7f800000)
"""
    NaN32

A not-a-number value of type [`Float32`](@ref).
"""
const NaN32 = bitcast(Float32, 0x7fc00000)
const Inf64 = bitcast(Float64, 0x7ff0000000000000)
const NaN64 = bitcast(Float64, 0x7ff8000000000000)

const Inf = Inf64
"""
    Inf, Inf64

Positive infinity of type [`Float64`](@ref).
"""
Inf, Inf64

const NaN = NaN64
"""
    NaN, NaN64

A not-a-number value of type [`Float64`](@ref).
"""
NaN, NaN64

## conversions to floating-point ##
function Float16(x::Integer)
    convert(Float16, convert(Float32, x))
end
for t in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval promote_rule(::Type{Float16}, ::Type{$t}) = Float16
end
function promote_rule(::Type{Float16}, ::Type{Bool})
    Float16
end

for t1 in (Float32, Float64)
    for st in (Int8, Int16, Int32, Int64)
        @eval begin
            (::Type{$t1})(x::($st)) = sitofp($t1, x)
            promote_rule(::Type{$t1}, ::Type{$st}) = $t1
        end
    end
    for ut in (Bool, UInt8, UInt16, UInt32, UInt64)
        @eval begin
            (::Type{$t1})(x::($ut)) = uitofp($t1, x)
            promote_rule(::Type{$t1}, ::Type{$ut}) = $t1
        end
    end
end
function (::Type{T})(x::Float16) where T <: Integer
    T(Float32(x))
end

function Bool(x::Real)
    if x == 0
        false
    else
        if x == 1
            true
        else
            throw(InexactError(:Bool, Bool, x))
        end
    end
end

function promote_rule(::Type{Float64}, ::Type{UInt128})
    Float64
end
function promote_rule(::Type{Float64}, ::Type{Int128})
    Float64
end
function promote_rule(::Type{Float32}, ::Type{UInt128})
    Float32
end
function promote_rule(::Type{Float32}, ::Type{Int128})
    Float32
end

function Float64(x::UInt128)
    x == 0 && return 0.0
    n = 128-leading_zeros(x) # ndigits0z(x,2)
    if n <= 53
        y = ((x % UInt64) << (53-n)) & 0x000f_ffff_ffff_ffff
    else
        y = ((x >> (n-54)) % UInt64) & 0x001f_ffff_ffff_ffff # keep 1 extra bit
        y = (y+1)>>1 # round, ties up (extra leading bit in case of next exponent)
        y &= ~UInt64(trailing_zeros(x) == (n-54)) # fix last bit to round to even
    end
    d = ((n+1022) % UInt64) << 52
    reinterpret(Float64, d + y)
end

function Float64(x::Int128)
    x == 0 && return 0.0
    s = ((x >>> 64) % UInt64) & 0x8000_0000_0000_0000 # sign bit
    x = abs(x) % UInt128
    n = 128-leading_zeros(x) # ndigits0z(x,2)
    if n <= 53
        y = ((x % UInt64) << (53-n)) & 0x000f_ffff_ffff_ffff
    else
        y = ((x >> (n-54)) % UInt64) & 0x001f_ffff_ffff_ffff # keep 1 extra bit
        y = (y+1)>>1 # round, ties up (extra leading bit in case of next exponent)
        y &= ~UInt64(trailing_zeros(x) == (n-54)) # fix last bit to round to even
    end
    d = ((n+1022) % UInt64) << 52
    reinterpret(Float64, s | d + y)
end

function Float32(x::UInt128)
    x == 0 && return 0f0
    n = 128-leading_zeros(x) # ndigits0z(x,2)
    if n <= 24
        y = ((x % UInt32) << (24-n)) & 0x007f_ffff
    else
        y = ((x >> (n-25)) % UInt32) & 0x00ff_ffff # keep 1 extra bit
        y = (y+one(UInt32))>>1 # round, ties up (extra leading bit in case of next exponent)
        y &= ~UInt32(trailing_zeros(x) == (n-25)) # fix last bit to round to even
    end
    d = ((n+126) % UInt32) << 23
    reinterpret(Float32, d + y)
end

function Float32(x::Int128)
    x == 0 && return 0f0
    s = ((x >>> 96) % UInt32) & 0x8000_0000 # sign bit
    x = abs(x) % UInt128
    n = 128-leading_zeros(x) # ndigits0z(x,2)
    if n <= 24
        y = ((x % UInt32) << (24-n)) & 0x007f_ffff
    else
        y = ((x >> (n-25)) % UInt32) & 0x00ff_ffff # keep 1 extra bit
        y = (y+one(UInt32))>>1 # round, ties up (extra leading bit in case of next exponent)
        y &= ~UInt32(trailing_zeros(x) == (n-25)) # fix last bit to round to even
    end
    d = ((n+126) % UInt32) << 23
    reinterpret(Float32, s | d + y)
end

function Float16(val::Float32)
    f = reinterpret(UInt32, val)
    if isnan(val)
        t = 0x8000 ⊻ (0x8000 & ((f >> 0x10) % UInt16))
        return reinterpret(Float16, t ⊻ ((f >> 0xd) % UInt16))
    end
    i = (f >> 23) & 0x1ff + 1
    @inbounds sh = shifttable[i]
    f &= 0x007fffff
    @inbounds h = (basetable[i] + (f >> sh)) % UInt16
    # round
    # NOTE: we maybe should ignore NaNs here, but the payload is
    # getting truncated anyway so "rounding" it might not matter
    nextbit = (f >> (sh-1)) & 1
    if nextbit != 0
        # Round halfway to even or check lower bits
        if h&1 == 1 || (f & ((1<<(sh-1))-1)) != 0
            h += UInt16(1)
        end
    end
    reinterpret(Float16, h)
end

function Float32(val::Float16)
    local ival::UInt32 = reinterpret(UInt16, val)
    local sign::UInt32 = (ival & 0x8000) >> 15
    local exp::UInt32  = (ival & 0x7c00) >> 10
    local sig::UInt32  = (ival & 0x3ff) >> 0
    local ret::UInt32

    if exp == 0
        if sig == 0
            sign = sign << 31
            ret = sign | exp | sig
        else
            n_bit = 1
            bit = 0x0200
            while (bit & sig) == 0
                n_bit = n_bit + 1
                bit = bit >> 1
            end
            sign = sign << 31
            exp = ((-14 - n_bit + 127) << 23) % UInt32
            sig = ((sig & (~bit)) << n_bit) << (23 - 10)
            ret = sign | exp | sig
        end
    elseif exp == 0x1f
        if sig == 0  # Inf
            if sign == 0
                ret = 0x7f800000
            else
                ret = 0xff800000
            end
        else  # NaN
            ret = 0x7fc00000 | (sign<<31) | (sig<<(23-10))
        end
    else
        sign = sign << 31
        exp  = ((exp - 15 + 127) << 23) % UInt32
        sig  = sig << (23 - 10)
        ret = sign | exp | sig
    end
    return reinterpret(Float32, ret)
end

# Float32 -> Float16 algorithm from:
#   "Fast Half Float Conversion" by Jeroen van der Zijp
#   ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf

let _basetable = Vector{UInt16}(undef, 512),
    _shifttable = Vector{UInt8}(undef, 512)
    for i = 0:255
        e = i - 127
        if e < -24  # Very small numbers map to zero
            _basetable[i|0x000+1] = 0x0000
            _basetable[i|0x100+1] = 0x8000
            _shifttable[i|0x000+1] = 24
            _shifttable[i|0x100+1] = 24
        elseif e < -14  # Small numbers map to denorms
            _basetable[i|0x000+1] = (0x0400>>(-e-14))
            _basetable[i|0x100+1] = (0x0400>>(-e-14)) | 0x8000
            _shifttable[i|0x000+1] = -e-1
            _shifttable[i|0x100+1] = -e-1
        elseif e <= 15  # Normal numbers just lose precision
            _basetable[i|0x000+1] = ((e+15)<<10)
            _basetable[i|0x100+1] = ((e+15)<<10) | 0x8000
            _shifttable[i|0x000+1] = 13
            _shifttable[i|0x100+1] = 13
        elseif e < 128  # Large numbers map to Infinity
            _basetable[i|0x000+1] = 0x7C00
            _basetable[i|0x100+1] = 0xFC00
            _shifttable[i|0x000+1] = 24
            _shifttable[i|0x100+1] = 24
        else  # Infinity and NaN's stay Infinity and NaN's
            _basetable[i|0x000+1] = 0x7C00
            _basetable[i|0x100+1] = 0xFC00
            _shifttable[i|0x000+1] = 13
            _shifttable[i|0x100+1] = 13
        end
    end
    global const shifttable = (_shifttable...,)
    global const basetable = (_basetable...,)
end

#convert(::Type{Float16}, x::Float32) = fptrunc(Float16, x)
function Float32(x::Float64)
    fptrunc(Float32, x)
end
function Float16(x::Float64)
    Float16(Float32(x))
end

#convert(::Type{Float32}, x::Float16) = fpext(Float32, x)
function Float64(x::Float32)
    fpext(Float64, x)
end
function Float64(x::Float16)
    Float64(Float32(x))
end

function AbstractFloat(x::Bool)
    Float64(x)
end
function AbstractFloat(x::Int8)
    Float64(x)
end
function AbstractFloat(x::Int16)
    Float64(x)
end
function AbstractFloat(x::Int32)
    Float64(x)
end
function AbstractFloat(x::Int64)
    Float64(x)
end # LOSSY
function AbstractFloat(x::Int128)
    Float64(x)
end # LOSSY
function AbstractFloat(x::UInt8)
    Float64(x)
end
function AbstractFloat(x::UInt16)
    Float64(x)
end
function AbstractFloat(x::UInt32)
    Float64(x)
end
function AbstractFloat(x::UInt64)
    Float64(x)
end # LOSSY
function AbstractFloat(x::UInt128)
    Float64(x)
end # LOSSY

function Bool(x::Float16)
    if x == 0
        false
    else
        if x == 1
            true
        else
            throw(InexactError(:Bool, Bool, x))
        end
    end
end

"""
    float(x)

Convert a number or array to a floating point data type.
"""
float(x) = AbstractFloat(x)

"""
    float(T::Type)

Return an appropriate type to represent a value of type `T` as a floating point value.
Equivalent to `typeof(float(zero(T)))`.

# Examples
```jldoctest
julia> float(Complex{Int})
Complex{Float64}

julia> float(Int)
Float64
```
"""
float(::Type{T}) where {T<:Number} = typeof(float(zero(T)))
function float(::Type{T}) where T <: AbstractFloat
    T
end

"""
    unsafe_trunc(T, x)

Return the nearest integral value of type `T` whose absolute value is
less than or equal to `x`. If the value is not representable by `T`, an arbitrary value will
be returned.
"""
function unsafe_trunc end

for Ti in (Int8, Int16, Int32, Int64)
    @eval begin
        unsafe_trunc(::Type{$Ti}, x::Float16) = unsafe_trunc($Ti, Float32(x))
        unsafe_trunc(::Type{$Ti}, x::Float32) = fptosi($Ti, x)
        unsafe_trunc(::Type{$Ti}, x::Float64) = fptosi($Ti, x)
    end
end
for Ti in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        unsafe_trunc(::Type{$Ti}, x::Float16) = unsafe_trunc($Ti, Float32(x))
        unsafe_trunc(::Type{$Ti}, x::Float32) = fptoui($Ti, x)
        unsafe_trunc(::Type{$Ti}, x::Float64) = fptoui($Ti, x)
    end
end

function unsafe_trunc(::Type{UInt128}, x::Float64)
    xu = reinterpret(UInt64,x)
    k = Int(xu >> 52) & 0x07ff - 1075
    xu = (xu & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
    if k <= 0
        UInt128(xu >> -k)
    else
        UInt128(xu) << k
    end
end
function unsafe_trunc(::Type{Int128}, x::Float64)
    copysign(unsafe_trunc(UInt128,x) % Int128, x)
end

function unsafe_trunc(::Type{UInt128}, x::Float32)
    xu = reinterpret(UInt32,x)
    k = Int(xu >> 23) & 0x00ff - 150
    xu = (xu & 0x007f_ffff) | 0x0080_0000
    if k <= 0
        UInt128(xu >> -k)
    else
        UInt128(xu) << k
    end
end
function unsafe_trunc(::Type{Int128}, x::Float32)
    copysign(unsafe_trunc(UInt128,x) % Int128, x)
end

function unsafe_trunc(::Type{UInt128}, x::Float16)
    unsafe_trunc(UInt128, Float32(x))
end
function unsafe_trunc(::Type{Int128}, x::Float16)
    unsafe_trunc(Int128, Float32(x))
end

# matches convert methods
# also determines floor, ceil, round
function trunc(::Type{Signed}, x::Float32)
    trunc(Int, x)
end
function trunc(::Type{Signed}, x::Float64)
    trunc(Int, x)
end
function trunc(::Type{Unsigned}, x::Float32)
    trunc(UInt, x)
end
function trunc(::Type{Unsigned}, x::Float64)
    trunc(UInt, x)
end
function trunc(::Type{Integer}, x::Float32)
    trunc(Int, x)
end
function trunc(::Type{Integer}, x::Float64)
    trunc(Int, x)
end
function trunc(::Type{T}, x::Float16) where T <: Integer
    trunc(T, Float32(x))
end

# fallbacks
function floor(::Type{T}, x::AbstractFloat) where T <: Integer
    trunc(T, round(x, RoundDown))
end
function floor(::Type{T}, x::Float16) where T <: Integer
    floor(T, Float32(x))
end
function ceil(::Type{T}, x::AbstractFloat) where T <: Integer
    trunc(T, round(x, RoundUp))
end
function ceil(::Type{T}, x::Float16) where T <: Integer
    ceil(T, Float32(x))
end
function round(::Type{T}, x::AbstractFloat) where T <: Integer
    trunc(T, round(x, RoundNearest))
end
function round(::Type{T}, x::Float16) where T <: Integer
    round(T, Float32(x))
end

function round(x::Float64, r::RoundingMode{:ToZero})
    trunc_llvm(x)
end
function round(x::Float32, r::RoundingMode{:ToZero})
    trunc_llvm(x)
end
function round(x::Float64, r::RoundingMode{:Down})
    floor_llvm(x)
end
function round(x::Float32, r::RoundingMode{:Down})
    floor_llvm(x)
end
function round(x::Float64, r::RoundingMode{:Up})
    ceil_llvm(x)
end
function round(x::Float32, r::RoundingMode{:Up})
    ceil_llvm(x)
end
function round(x::Float64, r::RoundingMode{:Nearest})
    rint_llvm(x)
end
function round(x::Float32, r::RoundingMode{:Nearest})
    rint_llvm(x)
end

function round(x::Float16, r::RoundingMode{:ToZero})
    Float16(round(Float32(x), r))
end
function round(x::Float16, r::RoundingMode{:Down})
    Float16(round(Float32(x), r))
end
function round(x::Float16, r::RoundingMode{:Up})
    Float16(round(Float32(x), r))
end
function round(x::Float16, r::RoundingMode{:Nearest})
    Float16(round(Float32(x), r))
end

## floating point promotions ##
function promote_rule(::Type{Float32}, ::Type{Float16})
    Float32
end
function promote_rule(::Type{Float64}, ::Type{Float16})
    Float64
end
function promote_rule(::Type{Float64}, ::Type{Float32})
    Float64
end

function widen(::Type{Float16})
    Float32
end
function widen(::Type{Float32})
    Float64
end

## floating point arithmetic ##
function -(x::Float64)
    neg_float(x)
end
function -(x::Float32)
    neg_float(x)
end
function -(x::Float16)
    reinterpret(Float16, reinterpret(UInt16, x) ⊻ 0x8000)
end

for op in (:+, :-, :*, :/, :\, :^)
    @eval ($op)(a::Float16, b::Float16) = Float16(($op)(Float32(a), Float32(b)))
end
function +(x::Float32, y::Float32)
    add_float(x, y)
end
function +(x::Float64, y::Float64)
    add_float(x, y)
end
function -(x::Float32, y::Float32)
    sub_float(x, y)
end
function -(x::Float64, y::Float64)
    sub_float(x, y)
end
function *(x::Float32, y::Float32)
    mul_float(x, y)
end
function *(x::Float64, y::Float64)
    mul_float(x, y)
end
function /(x::Float32, y::Float32)
    div_float(x, y)
end
function /(x::Float64, y::Float64)
    div_float(x, y)
end

function muladd(x::Float32, y::Float32, z::Float32)
    muladd_float(x, y, z)
end
function muladd(x::Float64, y::Float64, z::Float64)
    muladd_float(x, y, z)
end
function muladd(a::Float16, b::Float16, c::Float16)
    Float16(muladd(Float32(a), Float32(b), Float32(c)))
end

# TODO: faster floating point div?
# TODO: faster floating point fld?
# TODO: faster floating point mod?

for func in (:div,:fld,:cld,:rem,:mod)
    @eval begin
        $func(a::Float16,b::Float16) = Float16($func(Float32(a),Float32(b)))
    end
end

function rem(x::Float32, y::Float32)
    rem_float(x, y)
end
function rem(x::Float64, y::Float64)
    rem_float(x, y)
end

function cld(x::T, y::T) where T <: AbstractFloat
    -(fld(-x, y))
end

function mod(x::T, y::T) where T<:AbstractFloat
    r = rem(x,y)
    if r == 0
        copysign(r,y)
    elseif (r > 0) ⊻ (y > 0)
        r+y
    else
        r
    end
end

## floating point comparisons ##
function ==(x::Float16, y::Float16)
    ix = reinterpret(UInt16,x)
    iy = reinterpret(UInt16,y)
    if (ix|iy)&0x7fff > 0x7c00 #isnan(x) || isnan(y)
        return false
    end
    if (ix|iy)&0x7fff == 0x0000
        return true
    end
    return ix == iy
end
function ==(x::Float32, y::Float32)
    eq_float(x, y)
end
function ==(x::Float64, y::Float64)
    eq_float(x, y)
end
function !=(x::Float32, y::Float32)
    ne_float(x, y)
end
function !=(x::Float64, y::Float64)
    ne_float(x, y)
end
function <(x::Float32, y::Float32)
    lt_float(x, y)
end
function <(x::Float64, y::Float64)
    lt_float(x, y)
end
function <=(x::Float32, y::Float32)
    le_float(x, y)
end
function <=(x::Float64, y::Float64)
    le_float(x, y)
end

function isequal(x::Float32, y::Float32)
    fpiseq(x, y)
end
function isequal(x::Float64, y::Float64)
    fpiseq(x, y)
end
function isless(x::Float32, y::Float32)
    fpislt(x, y)
end
function isless(x::Float64, y::Float64)
    fpislt(x, y)
end
for op in (:<, :<=, :isless)
    @eval ($op)(a::Float16, b::Float16) = ($op)(Float32(a), Float32(b))
end

# Exact Float (Tf) vs Integer (Ti) comparisons
# Assumes:
# - typemax(Ti) == 2^n-1
# - typemax(Ti) can't be exactly represented by Tf:
#   => Tf(typemax(Ti)) == 2^n or Inf
# - typemin(Ti) can be exactly represented by Tf
#
# 1. convert y::Ti to float fy::Tf
# 2. perform Tf comparison x vs fy
# 3. if x == fy, check if (1) resulted in rounding:
#  a. convert fy back to Ti and compare with original y
#  b. unsafe_convert undefined behaviour if fy == Tf(typemax(Ti))
#     (but consequently x == fy > y)
for Ti in (Int64,UInt64,Int128,UInt128)
    for Tf in (Float32,Float64)
        @eval begin
            function ==(x::$Tf, y::$Ti)
                fy = ($Tf)(y)
                (x == fy) & (fy != $(Tf(typemax(Ti)))) & (y == unsafe_trunc($Ti,fy))
            end
            ==(y::$Ti, x::$Tf) = x==y

            function <(x::$Ti, y::$Tf)
                fx = ($Tf)(x)
                (fx < y) | ((fx == y) & ((fx == $(Tf(typemax(Ti)))) | (x < unsafe_trunc($Ti,fx)) ))
            end
            function <=(x::$Ti, y::$Tf)
                fx = ($Tf)(x)
                (fx < y) | ((fx == y) & ((fx == $(Tf(typemax(Ti)))) | (x <= unsafe_trunc($Ti,fx)) ))
            end

            function <(x::$Tf, y::$Ti)
                fy = ($Tf)(y)
                (x < fy) | ((x == fy) & (fy < $(Tf(typemax(Ti)))) & (unsafe_trunc($Ti,fy) < y))
            end
            function <=(x::$Tf, y::$Ti)
                fy = ($Tf)(y)
                (x < fy) | ((x == fy) & (fy < $(Tf(typemax(Ti)))) & (unsafe_trunc($Ti,fy) <= y))
            end
        end
    end
end
for op in (:(==), :<, :<=)
    @eval begin
        ($op)(x::Float16, y::Union{Int128,UInt128,Int64,UInt64}) = ($op)(Float64(x), Float64(y))
        ($op)(x::Union{Int128,UInt128,Int64,UInt64}, y::Float16) = ($op)(Float64(x), Float64(y))

        ($op)(x::Union{Float16,Float32}, y::Union{Int32,UInt32}) = ($op)(Float64(x), Float64(y))
        ($op)(x::Union{Int32,UInt32}, y::Union{Float16,Float32}) = ($op)(Float64(x), Float64(y))

        ($op)(x::Float16, y::Union{Int16,UInt16}) = ($op)(Float32(x), Float32(y))
        ($op)(x::Union{Int16,UInt16}, y::Float16) = ($op)(Float32(x), Float32(y))
    end
end


function abs(x::Float16)
    reinterpret(Float16, reinterpret(UInt16, x) & 0x7fff)
end
function abs(x::Float32)
    abs_float(x)
end
function abs(x::Float64)
    abs_float(x)
end

"""
    isnan(f) -> Bool

Test whether a floating point number is not a number (NaN).
"""
isnan(x::AbstractFloat) = x != x
function isnan(x::Float16)
    reinterpret(UInt16, x) & 0x7fff > 0x7c00
end
function isnan(x::Real)
    false
end

"""
    isfinite(f) -> Bool

Test whether a number is finite.

# Examples
```jldoctest
julia> isfinite(5)
true

julia> isfinite(NaN32)
false
```
"""
isfinite(x::AbstractFloat) = x - x == 0
function isfinite(x::Float16)
    reinterpret(UInt16, x) & 0x7c00 != 0x7c00
end
function isfinite(x::Real)
    (decompose(x))[3] != 0
end
function isfinite(x::Integer)
    true
end

"""
    isinf(f) -> Bool

Test whether a number is infinite.
"""
isinf(x::Real) = !isnan(x) & !isfinite(x)

## hashing small, built-in numeric types ##

function hx(a::UInt64, b::Float64, h::UInt)
    hash_uint64((3a + reinterpret(UInt64, b)) - h)
end
const hx_NaN = hx(UInt64(0), NaN, UInt(0  ))

function hash(x::UInt64, h::UInt)
    hx(x, Float64(x), h)
end
function hash(x::Int64, h::UInt)
    hx(reinterpret(UInt64, abs(x)), Float64(x), h)
end
function hash(x::Float64, h::UInt)
    if isnan(x)
        hx_NaN ⊻ h
    else
        hx(fptoui(UInt64, abs(x)), x, h)
    end
end

function hash(x::Union{Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32}, h::UInt)
    hash(Int64(x), h)
end
function hash(x::Float32, h::UInt)
    hash(Float64(x), h)
end

"""
    precision(num::AbstractFloat)

Get the precision of a floating point number, as defined by the effective number of bits in
the mantissa.
"""
function precision end

function precision(::Type{Float16})
    11
end
function precision(::Type{Float32})
    24
end
function precision(::Type{Float64})
    53
end
function precision(::T) where T <: AbstractFloat
    precision(T)
end

"""
    uabs(x::Integer)

Return the absolute value of `x`, possibly returning a different type should the
operation be susceptible to overflow. This typically arises when `x` is a two's complement
signed integer, so that `abs(typemin(x)) == typemin(x) < 0`, in which case the result of
`uabs(x)` will be an unsigned integer of the same size.
"""
uabs(x::Integer) = abs(x)
function uabs(x::BitSigned)
    unsigned(abs(x))
end


"""
    nextfloat(x::IEEEFloat, n::Integer)

The result of `n` iterative applications of `nextfloat` to `x` if `n >= 0`, or `-n`
applications of `prevfloat` if `n < 0`.
"""
function nextfloat(f::IEEEFloat, d::Integer)
    F = typeof(f)
    fumax = reinterpret(Unsigned, F(Inf))
    U = typeof(fumax)

    isnan(f) && return f
    fi = reinterpret(Signed, f)
    fneg = fi < 0
    fu = unsigned(fi & typemax(fi))

    dneg = d < 0
    da = uabs(d)
    if da > typemax(U)
        fneg = dneg
        fu = fumax
    else
        du = da % U
        if fneg ⊻ dneg
            if du > fu
                fu = min(fumax, du - fu)
                fneg = !fneg
            else
                fu = fu - du
            end
        else
            if fumax - fu < du
                fu = fumax
            else
                fu = fu + du
            end
        end
    end
    if fneg
        fu |= sign_mask(F)
    end
    reinterpret(F, fu)
end

"""
    nextfloat(x::AbstractFloat)

Return the smallest floating point number `y` of the same type as `x` such `x < y`. If no
such `y` exists (e.g. if `x` is `Inf` or `NaN`), then return `x`.
"""
nextfloat(x::AbstractFloat) = nextfloat(x,1)

"""
    prevfloat(x::AbstractFloat, n::Integer)

The result of `n` iterative applications of `prevfloat` to `x` if `n >= 0`, or `-n`
applications of `nextfloat` if `n < 0`.
"""
prevfloat(x::AbstractFloat, d::Integer) = nextfloat(x, -d)

"""
    prevfloat(x::AbstractFloat)

Return the largest floating point number `y` of the same type as `x` such `y < x`. If no
such `y` exists (e.g. if `x` is `-Inf` or `NaN`), then return `x`.
"""
prevfloat(x::AbstractFloat) = nextfloat(x,-1)

for Ti in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    for Tf in (Float32, Float64)
        if Ti <: Unsigned || sizeof(Ti) < sizeof(Tf)
            # Here `Tf(typemin(Ti))-1` is exact, so we can compare the lower-bound
            # directly. `Tf(typemax(Ti))+1` is either always exactly representable, or
            # rounded to `Inf` (e.g. when `Ti==UInt128 && Tf==Float32`).
            @eval begin
                function trunc(::Type{$Ti},x::$Tf)
                    if $(Tf(typemin(Ti))-one(Tf)) < x < $(Tf(typemax(Ti))+one(Tf))
                        return unsafe_trunc($Ti,x)
                    else
                        throw(InexactError(:trunc, $Ti, x))
                    end
                end
                function (::Type{$Ti})(x::$Tf)
                    if ($(Tf(typemin(Ti))) <= x <= $(Tf(typemax(Ti)))) && (round(x, RoundToZero) == x)
                        return unsafe_trunc($Ti,x)
                    else
                        throw(InexactError($(Expr(:quote,Ti.name.name)), $Ti, x))
                    end
                end
            end
        else
            # Here `eps(Tf(typemin(Ti))) > 1`, so the only value which can be truncated to
            # `Tf(typemin(Ti)` is itself. Similarly, `Tf(typemax(Ti))` is inexact and will
            # be rounded up. This assumes that `Tf(typemin(Ti)) > -Inf`, which is true for
            # these types, but not for `Float16` or larger integer types.
            @eval begin
                function trunc(::Type{$Ti},x::$Tf)
                    if $(Tf(typemin(Ti))) <= x < $(Tf(typemax(Ti)))
                        return unsafe_trunc($Ti,x)
                    else
                        throw(InexactError(:trunc, $Ti, x))
                    end
                end
                function (::Type{$Ti})(x::$Tf)
                    if ($(Tf(typemin(Ti))) <= x < $(Tf(typemax(Ti)))) && (round(x, RoundToZero) == x)
                        return unsafe_trunc($Ti,x)
                    else
                        throw(InexactError($(Expr(:quote,Ti.name.name)), $Ti, x))
                    end
                end
            end
        end
    end
end

"""
    issubnormal(f) -> Bool

Test whether a floating point number is subnormal.
"""
function issubnormal(x::T) where {T<:IEEEFloat}
    y = reinterpret(Unsigned, x)
    (y & exponent_mask(T) == 0) & (y & significand_mask(T) != 0)
end

@eval begin
    typemin(::Type{Float16}) = $(bitcast(Float16, 0xfc00))
    typemax(::Type{Float16}) = $(Inf16)
    typemin(::Type{Float32}) = $(-Inf32)
    typemax(::Type{Float32}) = $(Inf32)
    typemin(::Type{Float64}) = $(-Inf64)
    typemax(::Type{Float64}) = $(Inf64)
    typemin(x::T) where {T<:Real} = typemin(T)
    typemax(x::T) where {T<:Real} = typemax(T)

    floatmin(::Type{Float16}) = $(bitcast(Float16, 0x0400))
    floatmin(::Type{Float32}) = $(bitcast(Float32, 0x00800000))
    floatmin(::Type{Float64}) = $(bitcast(Float64, 0x0010000000000000))
    floatmax(::Type{Float16}) = $(bitcast(Float16, 0x7bff))
    floatmax(::Type{Float32}) = $(bitcast(Float32, 0x7f7fffff))
    floatmax(::Type{Float64}) = $(bitcast(Float64, 0x7fefffffffffffff))

    eps(x::AbstractFloat) = isfinite(x) ? abs(x) >= floatmin(x) ? ldexp(eps(typeof(x)), exponent(x)) : nextfloat(zero(x)) : oftype(x, NaN)
    eps(::Type{Float16}) = $(bitcast(Float16, 0x1400))
    eps(::Type{Float32}) = $(bitcast(Float32, 0x34000000))
    eps(::Type{Float64}) = $(bitcast(Float64, 0x3cb0000000000000))
    eps() = eps(Float64)
end

"""
    floatmin(T)

The smallest in absolute value non-subnormal value representable by the given
floating-point DataType `T`.
"""
floatmin(x::T) where {T<:AbstractFloat} = floatmin(T)

"""
    floatmax(T)

The highest finite value representable by the given floating-point DataType `T`.

# Examples
```jldoctest
julia> floatmax(Float16)
Float16(6.55e4)

julia> floatmax(Float32)
3.4028235f38
```
"""
floatmax(x::T) where {T<:AbstractFloat} = floatmax(T)

function floatmin()
    floatmin(Float64)
end
function floatmax()
    floatmax(Float64)
end

"""
    eps(::Type{T}) where T<:AbstractFloat
    eps()

Return the *machine epsilon* of the floating point type `T` (`T = Float64` by
default). This is defined as the gap between 1 and the next largest value representable by
`typeof(one(T))`, and is equivalent to `eps(one(T))`.  (Since `eps(T)` is a
bound on the *relative error* of `T`, it is a "dimensionless" quantity like [`one`](@ref).)

# Examples
```jldoctest
julia> eps()
2.220446049250313e-16

julia> eps(Float32)
1.1920929f-7

julia> 1.0 + eps()
1.0000000000000002

julia> 1.0 + eps()/2
1.0
```
"""
eps(::Type{<:AbstractFloat})

"""
    eps(x::AbstractFloat)

Return the *unit in last place* (ulp) of `x`. This is the distance between consecutive
representable floating point values at `x`. In most cases, if the distance on either side
of `x` is different, then the larger of the two is taken, that is

    eps(x) == max(x-prevfloat(x), nextfloat(x)-x)

The exceptions to this rule are the smallest and largest finite values
(e.g. `nextfloat(-Inf)` and `prevfloat(Inf)` for [`Float64`](@ref)), which round to the
smaller of the values.

The rationale for this behavior is that `eps` bounds the floating point rounding
error. Under the default `RoundNearest` rounding mode, if ``y`` is a real number and ``x``
is the nearest floating point number to ``y``, then

```math
|y-x| \\leq \\operatorname{eps}(x)/2.
```

# Examples
```jldoctest
julia> eps(1.0)
2.220446049250313e-16

julia> eps(prevfloat(2.0))
2.220446049250313e-16

julia> eps(2.0)
4.440892098500626e-16

julia> x = prevfloat(Inf)      # largest finite Float64
1.7976931348623157e308

julia> x + eps(x)/2            # rounds up
Inf

julia> x + prevfloat(eps(x)/2) # rounds down
1.7976931348623157e308
```
"""
eps(::AbstractFloat)


## byte order swaps for arbitrary-endianness serialization/deserialization ##
function bswap(x::IEEEFloat)
    bswap_int(x)
end

# bit patterns
function reinterpret(::Type{Unsigned}, x::Float64)
    reinterpret(UInt64, x)
end
function reinterpret(::Type{Unsigned}, x::Float32)
    reinterpret(UInt32, x)
end
function reinterpret(::Type{Signed}, x::Float64)
    reinterpret(Int64, x)
end
function reinterpret(::Type{Signed}, x::Float32)
    reinterpret(Int32, x)
end

function sign_mask(::Type{Float64})
    0x8000000000000000
end
function exponent_mask(::Type{Float64})
    0x7ff0000000000000
end
function exponent_one(::Type{Float64})
    0x3ff0000000000000
end
function exponent_half(::Type{Float64})
    0x3fe0000000000000
end
function significand_mask(::Type{Float64})
    0x000fffffffffffff
end

function sign_mask(::Type{Float32})
    0x80000000
end
function exponent_mask(::Type{Float32})
    0x7f800000
end
function exponent_one(::Type{Float32})
    0x3f800000
end
function exponent_half(::Type{Float32})
    0x3f000000
end
function significand_mask(::Type{Float32})
    0x007fffff
end

function sign_mask(::Type{Float16})
    0x8000
end
function exponent_mask(::Type{Float16})
    0x7c00
end
function exponent_one(::Type{Float16})
    0x3c00
end
function exponent_half(::Type{Float16})
    0x3800
end
function significand_mask(::Type{Float16})
    0x03ff
end

# integer size of float
function uinttype(::Type{Float64})
    UInt64
end
function uinttype(::Type{Float32})
    UInt32
end
function uinttype(::Type{Float16})
    UInt16
end

function Base.iszero(x::Float16)
    reinterpret(UInt16, x) & ~(sign_mask(Float16)) == 0x0000
end

## Array operations on floating point numbers ##

function float(A::AbstractArray{<:AbstractFloat})
    A
end

function float(A::AbstractArray{T}) where T
    if !isconcretetype(T)
        error("`float` not defined on abstractly-typed arrays; please convert to a more specific type")
    end
    convert(AbstractArray{typeof(float(zero(T)))}, A)
end

function float(r::StepRange)
    float(r.start):float(r.step):float(last(r))
end
function float(r::UnitRange)
    float(r.start):float(last(r))
end
function float(r::StepRangeLen{T}) where T
    StepRangeLen{typeof(float(T(r.ref)))}(float(r.ref), float(r.step), length(r), r.offset)
end
function float(r::LinRange)
    LinRange(float(r.start), float(r.stop), length(r))
end
