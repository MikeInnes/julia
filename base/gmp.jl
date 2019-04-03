# This file is a part of Julia. License is MIT: https://julialang.org/license

module GMP

export BigInt

import .Base: *, +, -, /, <, <<, >>, >>>, <=, ==, >, >=, ^, (~), (&), (|), xor,
             binomial, cmp, convert, div, divrem, factorial, fld, gcd, gcdx, lcm, mod,
             ndigits, promote_rule, rem, show, isqrt, string, powermod,
             sum, trailing_zeros, trailing_ones, count_ones, tryparse_internal,
             bin, oct, dec, hex, isequal, invmod, _prevpow2, _nextpow2, ndigits0zpb,
             widen, signed, unsafe_trunc, trunc, iszero, isone, big, flipsign, signbit,
             hastypemax

if Clong == Int32
    const ClongMax = Union{Int8, Int16, Int32}
    const CulongMax = Union{UInt8, UInt16, UInt32}
else
    const ClongMax = Union{Int8, Int16, Int32, Int64}
    const CulongMax = Union{UInt8, UInt16, UInt32, UInt64}
end
const CdoubleMax = Union{Float16, Float32, Float64}

function version()
    VersionNumber(unsafe_string(unsafe_load(cglobal((:__gmp_version, :libgmp), Ptr{Cchar}))))
end
function bits_per_limb()
    Int(unsafe_load(cglobal((:__gmp_bits_per_limb, :libgmp), Cint)))
end

const VERSION = version()
const BITS_PER_LIMB = bits_per_limb()

# GMP's mp_limb_t is by default a typedef of `unsigned long`, but can also be configured to be either
# `unsigned int` or `unsigned long long int`. The correct unsigned type is here named Limb, and must
# be used whenever mp_limb_t is in the signature of ccall'ed GMP functions.
if BITS_PER_LIMB == 32
    const Limb = UInt32
    const SLimbMax = Union{Int8, Int16, Int32}
    const ULimbMax = Union{UInt8, UInt16, UInt32}
elseif BITS_PER_LIMB == 64
    const Limb = UInt64
    const SLimbMax = Union{Int8, Int16, Int32, Int64}
    const ULimbMax = Union{UInt8, UInt16, UInt32, UInt64}
else
    error("GMP: cannot determine the type mp_limb_t (__gmp_bits_per_limb == $BITS_PER_LIMB)")
end

"""
    BigInt <: Signed

Arbitrary precision integer type.
"""
mutable struct BigInt <: Signed
    alloc::Cint
    size::Cint
    d::Ptr{Limb}

    function BigInt()
        b = new(zero(Cint), zero(Cint), C_NULL)
        MPZ.init!(b)
        finalizer(cglobal((:__gmpz_clear, :libgmp)), b)
        return b
    end
end

"""
    BigInt(x)

Create an arbitrary precision integer. `x` may be an `Int` (or anything that can be
converted to an `Int`). The usual mathematical operators are defined for this type, and
results are promoted to a [`BigInt`](@ref).

Instances can be constructed from strings via [`parse`](@ref), or using the `big`
string literal.

# Examples
```jldoctest
julia> parse(BigInt, "42")
42

julia> big"313"
313
```
"""
BigInt(x)

"""
    ALLOC_OVERFLOW_FUNCTION

A reference that holds a boolean, if true, indicating julia is linked with a patched GMP that
does not abort on huge allocation and throws OutOfMemoryError instead.
"""
const ALLOC_OVERFLOW_FUNCTION = Ref(false)

function __init__()
    try
        if version().major != VERSION.major || bits_per_limb() != BITS_PER_LIMB
            msg = bits_per_limb() != BITS_PER_LIMB ? error : warn
            msg("The dynamically loaded GMP library (v\"$(version())\" with __gmp_bits_per_limb == $(bits_per_limb()))\n",
                "does not correspond to the compile time version (v\"$VERSION\" with __gmp_bits_per_limb == $BITS_PER_LIMB).\n",
                "Please rebuild Julia.")
        end

        ccall((:__gmp_set_memory_functions, :libgmp), Cvoid,
              (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),
              cglobal(:jl_gc_counted_malloc),
              cglobal(:jl_gc_counted_realloc_with_old_size),
              cglobal(:jl_gc_counted_free_with_size))
        ZERO.alloc, ZERO.size, ZERO.d = 0, 0, C_NULL
        ONE.alloc, ONE.size, ONE.d = 1, 1, pointer(_ONE)
    catch ex
        Base.showerror_nostdio(ex, "WARNING: Error during initialization of module GMP")
    end
    # This only works with a patched version of GMP, ignore otherwise
    try
        ccall((:__gmp_set_alloc_overflow_function, :libgmp), Cvoid,
              (Ptr{Cvoid},),
              cglobal(:jl_throw_out_of_memory_error))
        ALLOC_OVERFLOW_FUNCTION[] = true
    catch ex
        # ErrorException("ccall: could not find function...")
        if typeof(ex) != ErrorException
            rethrow()
        end
    end
end


module MPZ
# wrapping of libgmp functions
# - "output parameters" are labeled x, y, z, and are returned when appropriate
# - constant input parameters are labeled a, b, c
# - a method modifying its input has a "!" appendend to its name, according to Julia's conventions
# - some convenient methods are added (in addition to the pure MPZ ones), e.g. `add(a, b) = add!(BigInt(), a, b)`
#   and `add!(x, a) = add!(x, x, a)`.
using .Base.GMP: BigInt, Limb

const mpz_t = Ref{BigInt}
const bitcnt_t = Culong

function gmpz(op::Symbol)
    (Symbol(:__gmpz_, op), :libgmp)
end

function init!(x::BigInt)
    ccall((:__gmpz_init, :libgmp), Cvoid, (mpz_t,), x)
    x
end
function init2!(x::BigInt, a)
    ccall((:__gmpz_init2, :libgmp), Cvoid, (mpz_t, bitcnt_t), x, a)
    x
end

function realloc2!(x, a)
    ccall((:__gmpz_realloc2, :libgmp), Cvoid, (mpz_t, bitcnt_t), x, a)
    x
end
function realloc2(a)
    realloc2!(BigInt(), a)
end

function sizeinbase(a::BigInt, b)
    Int(ccall((:__gmpz_sizeinbase, :libgmp), Csize_t, (mpz_t, Cint), a, b))
end

for op in (:add, :sub, :mul, :fdiv_q, :tdiv_q, :fdiv_r, :tdiv_r, :gcd, :lcm, :and, :ior, :xor)
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a::BigInt, b::BigInt) = (ccall($(gmpz(op)), Cvoid, (mpz_t, mpz_t, mpz_t), x, a, b); x)
        $op(a::BigInt, b::BigInt) = $op!(BigInt(), a, b)
        $op!(x::BigInt, b::BigInt) = $op!(x, x, b)
    end
end

function invert!(x::BigInt, a::BigInt, b::BigInt)
    ccall((:__gmpz_invert, :libgmp), Cint, (mpz_t, mpz_t, mpz_t), x, a, b)
end
function invert(a::BigInt, b::BigInt)
    invert!(BigInt(), a, b)
end
function invert!(x::BigInt, b::BigInt)
    invert!(x, x, b)
end

for op in (:add_ui, :sub_ui, :mul_ui, :mul_2exp, :fdiv_q_2exp, :pow_ui, :bin_ui)
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a::BigInt, b) = (ccall($(gmpz(op)), Cvoid, (mpz_t, mpz_t, Culong), x, a, b); x)
        $op(a::BigInt, b) = $op!(BigInt(), a, b)
        $op!(x::BigInt, b) = $op!(x, x, b)
    end
end

function ui_sub!(x::BigInt, a, b::BigInt)
    ccall((:__gmpz_ui_sub, :libgmp), Cvoid, (mpz_t, Culong, mpz_t), x, a, b)
    x
end
function ui_sub(a, b::BigInt)
    ui_sub!(BigInt(), a, b)
end

for op in (:scan1, :scan0)
    @eval $op(a::BigInt, b) = Int(ccall($(gmpz(op)), Culong, (mpz_t, Culong), a, b))
end

function mul_si!(x::BigInt, a::BigInt, b)
    ccall((:__gmpz_mul_si, :libgmp), Cvoid, (mpz_t, mpz_t, Clong), x, a, b)
    x
end
function mul_si(a::BigInt, b)
    mul_si!(BigInt(), a, b)
end
function mul_si!(x::BigInt, b)
    mul_si!(x, x, b)
end

for op in (:neg, :com, :sqrt, :set)
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a::BigInt) = (ccall($(gmpz(op)), Cvoid, (mpz_t, mpz_t), x, a); x)
        $op(a::BigInt) = $op!(BigInt(), a)
    end
    op == :set && continue # MPZ.set!(x) would make no sense
    @eval $op!(x::BigInt) = $op!(x, x)
end

for (op, T) in ((:fac_ui, Culong), (:set_ui, Culong), (:set_si, Clong), (:set_d, Cdouble))
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a) = (ccall($(gmpz(op)), Cvoid, (mpz_t, $T), x, a); x)
        $op(a) = $op!(BigInt(), a)
    end
end

function popcount(a::BigInt)
    Int(ccall((:__gmpz_popcount, :libgmp), Culong, (mpz_t,), a))
end

function mpn_popcount(d::Ptr{Limb}, s::Integer)
    Int(ccall((:__gmpn_popcount, :libgmp), Culong, (Ptr{Limb}, Csize_t), d, s))
end
function mpn_popcount(a::BigInt)
    mpn_popcount(a.d, abs(a.size))
end

function tdiv_qr!(x::BigInt, y::BigInt, a::BigInt, b::BigInt)
    ccall((:__gmpz_tdiv_qr, :libgmp), Cvoid, (mpz_t, mpz_t, mpz_t, mpz_t), x, y, a, b)
    x, y
end
function tdiv_qr(a::BigInt, b::BigInt)
    tdiv_qr!(BigInt(), BigInt(), a, b)
end

function powm!(x::BigInt, a::BigInt, b::BigInt, c::BigInt)
    ccall((:__gmpz_powm, :libgmp), Cvoid, (mpz_t, mpz_t, mpz_t, mpz_t), x, a, b, c)
    x
end
function powm(a::BigInt, b::BigInt, c::BigInt)
    powm!(BigInt(), a, b, c)
end
function powm!(x::BigInt, b::BigInt, c::BigInt)
    powm!(x, x, b, c)
end

function gcdext!(x::BigInt, y::BigInt, z::BigInt, a::BigInt, b::BigInt)
    ccall((:__gmpz_gcdext, :libgmp), Cvoid, (mpz_t, mpz_t, mpz_t, mpz_t, mpz_t), x, y, z, a, b)
    x, y, z
end
function gcdext(a::BigInt, b::BigInt)
    gcdext!(BigInt(), BigInt(), BigInt(), a, b)
end

function cmp(a::BigInt, b::BigInt)
    Int(ccall((:__gmpz_cmp, :libgmp), Cint, (mpz_t, mpz_t), a, b))
end
function cmp_si(a::BigInt, b)
    Int(ccall((:__gmpz_cmp_si, :libgmp), Cint, (mpz_t, Clong), a, b))
end
function cmp_ui(a::BigInt, b)
    Int(ccall((:__gmpz_cmp_ui, :libgmp), Cint, (mpz_t, Culong), a, b))
end
function cmp_d(a::BigInt, b)
    Int(ccall((:__gmpz_cmp_d, :libgmp), Cint, (mpz_t, Cdouble), a, b))
end

function mpn_cmp(a::Ptr{Limb}, b::Ptr{Limb}, c)
    ccall((:__gmpn_cmp, :libgmp), Cint, (Ptr{Limb}, Ptr{Limb}, Clong), a, b, c)
end
function mpn_cmp(a::BigInt, b::BigInt, c)
    mpn_cmp(a.d, b.d, c)
end

function get_str!(x, a, b::BigInt)
    ccall((:__gmpz_get_str, :libgmp), Ptr{Cchar}, (Ptr{Cchar}, Cint, mpz_t), x, a, b)
    x
end
function set_str!(x::BigInt, a, b)
    Int(ccall((:__gmpz_set_str, :libgmp), Cint, (mpz_t, Ptr{UInt8}, Cint), x, a, b))
end
function get_d(a::BigInt)
    ccall((:__gmpz_get_d, :libgmp), Cdouble, (mpz_t,), a)
end

function limbs_write!(x::BigInt, a)
    ccall((:__gmpz_limbs_write, :libgmp), Ptr{Limb}, (mpz_t, Clong), x, a)
end
function limbs_finish!(x::BigInt, a)
    ccall((:__gmpz_limbs_finish, :libgmp), Cvoid, (mpz_t, Clong), x, a)
end
function import!(x::BigInt, a, b, c, d, e, f)
    ccall((:__gmpz_import, :libgmp), Cvoid, (mpz_t, Csize_t, Cint, Csize_t, Cint, Csize_t, Ptr{Cvoid}), x, a, b, c, d, e, f)
end

function setbit!(x, a)
    ccall((:__gmpz_setbit, :libgmp), Cvoid, (mpz_t, bitcnt_t), x, a)
    x
end
function tstbit(a::BigInt, b)
    ccall((:__gmpz_tstbit, :libgmp), Cint, (mpz_t, bitcnt_t), a, b) % Bool
end

end # module MPZ

const ZERO = BigInt()
const ONE  = BigInt()
const _ONE = Limb[1]

function widen(::Type{Int128})
    BigInt
end
function widen(::Type{UInt128})
    BigInt
end
function widen(::Type{BigInt})
    BigInt
end

function signed(x::BigInt)
    x
end

function BigInt(x::BigInt)
    x
end
function Signed(x::BigInt)
    x
end

function hastypemax(::Type{BigInt})
    false
end

function tryparse_internal(::Type{BigInt}, s::AbstractString, startpos::Int, endpos::Int, base_::Integer, raise::Bool)
    # don't make a copy in the common case where we are parsing a whole String
    bstr = startpos == firstindex(s) && endpos == lastindex(s) ? String(s) : String(SubString(s,startpos,endpos))

    sgn, base, i = Base.parseint_preamble(true,Int(base_),bstr,firstindex(bstr),lastindex(bstr))
    if !(2 <= base <= 62)
        raise && throw(ArgumentError("invalid base: base must be 2 ≤ base ≤ 62, got $base"))
        return nothing
    end
    if i == 0
        raise && throw(ArgumentError("premature end of integer: $(repr(bstr))"))
        return nothing
    end
    z = BigInt()
    if Base.containsnul(bstr)
        err = -1 # embedded NUL char (not handled correctly by GMP)
    else
        err = GC.@preserve bstr MPZ.set_str!(z, pointer(bstr)+(i-firstindex(bstr)), base)
    end
    if err != 0
        raise && throw(ArgumentError("invalid BigInt: $(repr(bstr))"))
        return nothing
    end
    flipsign!(z, sgn)
end

function BigInt(x::Union{Clong, Int32})
    MPZ.set_si(x)
end
function BigInt(x::Union{Culong, UInt32})
    MPZ.set_ui(x)
end
function BigInt(x::Bool)
    BigInt(UInt(x))
end

function unsafe_trunc(::Type{BigInt}, x::Union{Float32, Float64})
    MPZ.set_d(x)
end

function BigInt(x::Union{Float32,Float64})
    isinteger(x) || throw(InexactError(:BigInt, BigInt, x))
    unsafe_trunc(BigInt,x)
end

function trunc(::Type{BigInt}, x::Union{Float32,Float64})
    isfinite(x) || throw(InexactError(:trunc, BigInt, x))
    unsafe_trunc(BigInt,x)
end

function BigInt(x::Float16)
    BigInt(Float64(x))
end
function BigInt(x::Float32)
    BigInt(Float64(x))
end

function BigInt(x::Integer)
    x == 0 && return BigInt(Culong(0))
    nd = ndigits(x, base=2)
    z = MPZ.realloc2(nd)
    s = sign(x)
    s == -1 && (x = -x)
    x = unsigned(x)
    size = 0
    limbnbits = sizeof(Limb) << 3
    while nd > 0
        size += 1
        unsafe_store!(z.d, x % Limb, size)
        x >>>= limbnbits
        nd -= limbnbits
    end
    z.size = s*size
    z
end


function rem(x::BigInt, ::Type{Bool})
    (!(iszero(x)) & unsafe_load(x.d)) % Bool
end # never unsafe here

function rem(x::BigInt, ::Type{T}) where T <: Union{SLimbMax, ULimbMax}
    if iszero(x)
        zero(T)
    else
        flipsign(unsafe_load(x.d) % T, x.size)
    end
end

function rem(x::BigInt, ::Type{T}) where T<:Union{Base.BitUnsigned,Base.BitSigned}
    u = zero(T)
    for l = 1:min(abs(x.size), cld(sizeof(T), sizeof(Limb)))
        u += (unsafe_load(x.d, l) % T) << ((sizeof(Limb)<<3)*(l-1))
    end
    flipsign(u, x.size)
end

function rem(x::Integer, ::Type{BigInt})
    BigInt(x)
end

function (::Type{T})(x::BigInt) where T<:Base.BitUnsigned
    if sizeof(T) < sizeof(Limb)
        convert(T, convert(Limb,x))
    else
        0 <= x.size <= cld(sizeof(T),sizeof(Limb)) || throw(InexactError(nameof(T), T, x))
        x % T
    end
end

function (::Type{T})(x::BigInt) where T<:Base.BitSigned
    n = abs(x.size)
    if sizeof(T) < sizeof(Limb)
        SLimb = typeof(Signed(one(Limb)))
        convert(T, convert(SLimb, x))
    else
        0 <= n <= cld(sizeof(T),sizeof(Limb)) || throw(InexactError(nameof(T), T, x))
        y = x % T
        ispos(x) ⊻ (y > 0) && throw(InexactError(nameof(T), T, x)) # catch overflow
        y
    end
end


function Float64(n::BigInt, ::RoundingMode{:ToZero})
    MPZ.get_d(n)
end

function (::Type{T})(n::BigInt, ::RoundingMode{:ToZero}) where T<:Union{Float16,Float32}
    T(Float64(n,RoundToZero),RoundToZero)
end

function (::Type{T})(n::BigInt, ::RoundingMode{:Down}) where T<:CdoubleMax
    x = T(n,RoundToZero)
    x > n ? prevfloat(x) : x
end
function (::Type{T})(n::BigInt, ::RoundingMode{:Up}) where T<:CdoubleMax
    x = T(n,RoundToZero)
    x < n ? nextfloat(x) : x
end

function (::Type{T})(n::BigInt, ::RoundingMode{:Nearest}) where T<:CdoubleMax
    x = T(n,RoundToZero)
    if maxintfloat(T) <= abs(x) < T(Inf)
        r = n-BigInt(x)
        h = eps(x)/2
        if iseven(reinterpret(Unsigned,x)) # check if last bit is odd/even
            if r < -h
                return prevfloat(x)
            elseif r > h
                return nextfloat(x)
            end
        else
            if r <= -h
                return prevfloat(x)
            elseif r >= h
                return nextfloat(x)
            end
        end
    end
    x
end

function Float64(n::BigInt)
    Float64(n, RoundNearest)
end
function Float32(n::BigInt)
    Float32(n, RoundNearest)
end
function Float16(n::BigInt)
    Float16(n, RoundNearest)
end

function promote_rule(::Type{BigInt}, ::Type{<:Integer})
    BigInt
end

"""
    big(x)

Convert a number to a maximum precision representation (typically [`BigInt`](@ref) or
`BigFloat`). See [`BigFloat`](@ref) for information about some pitfalls with floating-point numbers.
"""
function big end

function big(::Type{<:Integer})
    BigInt
end
function big(::Type{<:Rational})
    Rational{BigInt}
end

function big(n::Integer)
    convert(BigInt, n)
end

# Binary ops
for (fJ, fC) in ((:+, :add), (:-,:sub), (:*, :mul),
                 (:fld, :fdiv_q), (:div, :tdiv_q), (:mod, :fdiv_r), (:rem, :tdiv_r),
                 (:gcd, :gcd), (:lcm, :lcm),
                 (:&, :and), (:|, :ior), (:xor, :xor))
    @eval begin
        ($fJ)(x::BigInt, y::BigInt) = MPZ.$fC(x, y)
    end
end

function /(x::BigInt, y::BigInt)
    float(x) / float(y)
end

function invmod(x::BigInt, y::BigInt)
    z = zero(BigInt)
    ya = abs(y)
    if ya == 1
        return z
    end
    if (y==0 || MPZ.invert!(z, x, ya) == 0)
        throw(DomainError(y))
    end
    # GMP always returns a positive inverse; we instead want to
    # normalize such that div(z, y) == 0, i.e. we want a negative z
    # when y is negative.
    if y < 0
        MPZ.add!(z, y)
    end
    # The postcondition is: mod(z * x, y) == mod(big(1), m) && div(z, y) == 0
    return z
end

# More efficient commutative operations
for (fJ, fC) in ((:+, :add), (:*, :mul), (:&, :and), (:|, :ior), (:xor, :xor))
    fC! = Symbol(fC, :!)
    @eval begin
        ($fJ)(a::BigInt, b::BigInt, c::BigInt) = MPZ.$fC!(MPZ.$fC(a, b), c)
        ($fJ)(a::BigInt, b::BigInt, c::BigInt, d::BigInt) = MPZ.$fC!(MPZ.$fC!(MPZ.$fC(a, b), c), d)
        ($fJ)(a::BigInt, b::BigInt, c::BigInt, d::BigInt, e::BigInt) =
            MPZ.$fC!(MPZ.$fC!(MPZ.$fC!(MPZ.$fC(a, b), c), d), e)
    end
end

# Basic arithmetic without promotion
function +(x::BigInt, c::CulongMax)
    MPZ.add_ui(x, c)
end
function +(c::CulongMax, x::BigInt)
    x + c
end

function -(x::BigInt, c::CulongMax)
    MPZ.sub_ui(x, c)
end
function -(c::CulongMax, x::BigInt)
    MPZ.ui_sub(c, x)
end

function +(x::BigInt, c::ClongMax)
    if c < 0
        x - -(c % Culong)
    else
        x + convert(Culong, c)
    end
end
function +(c::ClongMax, x::BigInt)
    if c < 0
        x - -(c % Culong)
    else
        x + convert(Culong, c)
    end
end
function -(x::BigInt, c::ClongMax)
    if c < 0
        x + -(c % Culong)
    else
        x - convert(Culong, c)
    end
end
function -(c::ClongMax, x::BigInt)
    if c < 0
        -((x + -(c % Culong)))
    else
        convert(Culong, c) - x
    end
end

function *(x::BigInt, c::CulongMax)
    MPZ.mul_ui(x, c)
end
function *(c::CulongMax, x::BigInt)
    x * c
end
function *(x::BigInt, c::ClongMax)
    MPZ.mul_si(x, c)
end
function *(c::ClongMax, x::BigInt)
    x * c
end

function /(x::BigInt, y::Union{ClongMax, CulongMax})
    float(x) / y
end
function /(x::Union{ClongMax, CulongMax}, y::BigInt)
    x / float(y)
end

# unary ops
function -(x::BigInt)
    MPZ.neg(x)
end
function ~(x::BigInt)
    MPZ.com(x)
end

function <<(x::BigInt, c::UInt)
    if c == 0
        x
    else
        MPZ.mul_2exp(x, c)
    end
end
function >>(x::BigInt, c::UInt)
    if c == 0
        x
    else
        MPZ.fdiv_q_2exp(x, c)
    end
end
function >>>(x::BigInt, c::UInt)
    x >> c
end

function trailing_zeros(x::BigInt)
    MPZ.scan1(x, 0)
end
function trailing_ones(x::BigInt)
    MPZ.scan0(x, 0)
end

function count_ones(x::BigInt)
    MPZ.popcount(x)
end

"""
    count_ones_abs(x::BigInt)

Number of ones in the binary representation of abs(x).
"""
count_ones_abs(x::BigInt) = iszero(x) ? 0 : MPZ.mpn_popcount(x)

function divrem(x::BigInt, y::BigInt)
    MPZ.tdiv_qr(x, y)
end

function cmp(x::BigInt, y::BigInt)
    sign(MPZ.cmp(x, y))
end
function cmp(x::BigInt, y::ClongMax)
    sign(MPZ.cmp_si(x, y))
end
function cmp(x::BigInt, y::CulongMax)
    sign(MPZ.cmp_ui(x, y))
end
function cmp(x::BigInt, y::Integer)
    cmp(x, big(y))
end
function cmp(x::Integer, y::BigInt)
    -(cmp(y, x))
end

function cmp(x::BigInt, y::CdoubleMax)
    if isnan(y)
        -1
    else
        sign(MPZ.cmp_d(x, y))
    end
end
function cmp(x::CdoubleMax, y::BigInt)
    -(cmp(y, x))
end

function isqrt(x::BigInt)
    MPZ.sqrt(x)
end

function ^(x::BigInt, y::Culong)
    MPZ.pow_ui(x, y)
end

function bigint_pow(x::BigInt, y::Integer)
    if y<0; throw(DomainError(y, "`y` cannot be negative.")); end
    @noinline throw1(y) =
        throw(OverflowError("exponent $y is too large and computation will overflow"))
    if x== 1; return x; end
    if x==-1; return isodd(y) ? x : -x; end
    if y>typemax(Culong)
       x==0 && return x

       #At this point, x is not 1, 0 or -1 and it is not possible to use
       #gmpz_pow_ui to compute the answer. Note that the magnitude of the
       #answer is:
       #- at least 2^(2^32-1) ≈ 10^(1.3e9) (if Culong === UInt32).
       #- at least 2^(2^64-1) ≈ 10^(5.5e18) (if Culong === UInt64).
       #
       #Assume that the answer will definitely overflow.

       throw1(y)
    end
    return x^convert(Culong, y)
end

function ^(x::BigInt, y::BigInt)
    bigint_pow(x, y)
end
function ^(x::BigInt, y::Bool)
    if y
        x
    else
        one(x)
    end
end
function ^(x::BigInt, y::Integer)
    bigint_pow(x, y)
end
function ^(x::Integer, y::BigInt)
    bigint_pow(BigInt(x), y)
end
function ^(x::Bool, y::BigInt)
    Base.power_by_squaring(x, y)
end

function powermod(x::BigInt, p::BigInt, m::BigInt)
    r = MPZ.powm(x, p, m)
    return m < 0 && r > 0 ? MPZ.add!(r, m) : r # choose sign consistent with mod(x^p, m)
end

function powermod(x::Integer, p::Integer, m::BigInt)
    powermod(big(x), big(p), m)
end

function gcdx(a::BigInt, b::BigInt)
    if iszero(b) # shortcut this to ensure consistent results with gcdx(a,b)
        return a < 0 ? (-a,-ONE,b) : (a,one(BigInt),b)
        # we don't return the globals ONE and ZERO in case the user wants to
        # mutate the result
    end
    g, s, t = MPZ.gcdext(a, b)
    if t == 0
        # work around a difference in some versions of GMP
        if a == b
            return g, t, s
        elseif abs(a)==abs(b)
            return g, t, -s
        end
    end
    g, s, t
end

function sum(arr::AbstractArray{BigInt})
    foldl(MPZ.add!, arr; init=BigInt(0))
end
# note: a similar implementation for `prod` won't be efficient:
# 1) the time complexity of the allocations is negligible compared to the multiplications
# 2) assuming arr contains similarly sized BigInts, the multiplications are much more
# performant when doing e.g. ((a1*a2)*(a2*a3))*(...) rather than a1*(a2*(a3*(...))),
# which is exactly what the default implementation of `prod` does, via mapreduce
# (which maybe could be slightly optimized for BigInt)

function factorial(x::BigInt)
    if isneg(x)
        BigInt(0)
    else
        MPZ.fac_ui(x)
    end
end

function binomial(n::BigInt, k::UInt)
    MPZ.bin_ui(n, k)
end
function binomial(n::BigInt, k::Integer)
    if k < 0
        BigInt(0)
    else
        binomial(n, UInt(k))
    end
end

function ==(x::BigInt, y::BigInt)
    cmp(x, y) == 0
end
function ==(x::BigInt, i::Integer)
    cmp(x, i) == 0
end
function ==(i::Integer, x::BigInt)
    cmp(x, i) == 0
end
function ==(x::BigInt, f::CdoubleMax)
    if isnan(f)
        false
    else
        cmp(x, f) == 0
    end
end
function ==(f::CdoubleMax, x::BigInt)
    if isnan(f)
        false
    else
        cmp(x, f) == 0
    end
end
function iszero(x::BigInt)
    x.size == 0
end
function isone(x::BigInt)
    x == Culong(1)
end

function <=(x::BigInt, y::BigInt)
    cmp(x, y) <= 0
end
function <=(x::BigInt, i::Integer)
    cmp(x, i) <= 0
end
function <=(i::Integer, x::BigInt)
    cmp(x, i) >= 0
end
function <=(x::BigInt, f::CdoubleMax)
    if isnan(f)
        false
    else
        cmp(x, f) <= 0
    end
end
function <=(f::CdoubleMax, x::BigInt)
    if isnan(f)
        false
    else
        cmp(x, f) >= 0
    end
end

function <(x::BigInt, y::BigInt)
    cmp(x, y) < 0
end
function <(x::BigInt, i::Integer)
    cmp(x, i) < 0
end
function <(i::Integer, x::BigInt)
    cmp(x, i) > 0
end
function <(x::BigInt, f::CdoubleMax)
    if isnan(f)
        false
    else
        cmp(x, f) < 0
    end
end
function <(f::CdoubleMax, x::BigInt)
    if isnan(f)
        false
    else
        cmp(x, f) > 0
    end
end
function isneg(x::BigInt)
    x.size < 0
end
function ispos(x::BigInt)
    x.size > 0
end

function signbit(x::BigInt)
    isneg(x)
end
function flipsign!(x::BigInt, y::Integer)
    signbit(y) && (x.size = -(x.size))
    x
end
function flipsign(x::BigInt, y::Integer)
    if signbit(y)
        -x
    else
        x
    end
end
function flipsign(x::BigInt, y::BigInt)
    if signbit(y)
        -x
    else
        x
    end
end
# above method to resolving ambiguities with flipsign(::T, ::T) where T<:Signed

function show(io::IO, x::BigInt)
    print(io, string(x))
end

function string(n::BigInt; base::Integer = 10, pad::Integer = 1)
    base < 0 && return Base._base(Int(base), n, pad, (base>0) & (n.size<0))
    2 <= base <= 62 || throw(ArgumentError("base must be 2 ≤ base ≤ 62, got $base"))
    iszero(n) && pad < 1 && return ""
    nd1 = ndigits(n, base=base)
    nd  = max(nd1, pad)
    sv  = Base.StringVector(nd + isneg(n))
    GC.@preserve sv MPZ.get_str!(pointer(sv) + nd - nd1, base, n)
    @inbounds for i = (1:nd-nd1) .+ isneg(n)
        sv[i] = '0' % UInt8
    end
    isneg(n) && (sv[1] = '-' % UInt8)
    String(sv)
end

function ndigits0zpb(x::BigInt, b::Integer)
    b < 2 && throw(DomainError(b, "`b` cannot be less than 2."))
    x.size == 0 && return 0 # for consistency with other ndigits0z methods
    if ispow2(b) && 2 <= b <= 62 # GMP assumes b is in this range
        MPZ.sizeinbase(x, b)
    else
        # non-base 2 mpz_sizeinbase might return an answer 1 too big
        # use property that log(b, x) < ndigits(x, base=b) <= log(b, x) + 1
        n = MPZ.sizeinbase(x, 2)
        lb = log2(b) # assumed accurate to <1ulp (true for openlibm)
        q,r = divrem(n,lb)
        iq = Int(q)
        maxerr = q*eps(lb) # maximum error in remainder
        if r-1.0 < maxerr
            abs(x) >= big(b)^iq ? iq+1 : iq
        elseif lb-r < maxerr
            abs(x) >= big(b)^(iq+1) ? iq+2 : iq+1
        else
            iq+1
        end
    end
end

# Fast paths for nextpow(2, x::BigInt)
# below, ONE is always left-shifted by at least one digit, so a new BigInt is
# allocated, which can be safely mutated
function _prevpow2(x::BigInt)
    if -2 <= x <= 2
        x
    else
        flipsign!(ONE << (ndigits(x, base=2) - 1), x)
    end
end
function _nextpow2(x::BigInt)
    if count_ones_abs(x) <= 1
        x
    else
        flipsign!(ONE << ndigits(x, base=2), x)
    end
end

function Base.checked_abs(x::BigInt)
    abs(x)
end
function Base.checked_neg(x::BigInt)
    -x
end
function Base.checked_add(a::BigInt, b::BigInt)
    a + b
end
function Base.checked_sub(a::BigInt, b::BigInt)
    a - b
end
function Base.checked_mul(a::BigInt, b::BigInt)
    a * b
end
function Base.checked_div(a::BigInt, b::BigInt)
    div(a, b)
end
function Base.checked_rem(a::BigInt, b::BigInt)
    rem(a, b)
end
function Base.checked_fld(a::BigInt, b::BigInt)
    fld(a, b)
end
function Base.checked_mod(a::BigInt, b::BigInt)
    mod(a, b)
end
function Base.checked_cld(a::BigInt, b::BigInt)
    cld(a, b)
end
function Base.add_with_overflow(a::BigInt, b::BigInt)
    (a + b, false)
end
function Base.sub_with_overflow(a::BigInt, b::BigInt)
    (a - b, false)
end
function Base.mul_with_overflow(a::BigInt, b::BigInt)
    (a * b, false)
end

function Base.deepcopy_internal(x::BigInt, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = MPZ.set(x)
    stackdict[x] = y
    return y
end

end # module
