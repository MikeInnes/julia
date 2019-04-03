# This file is a part of Julia. License is MIT: https://julialang.org/license

# Missing, missing and ismissing are defined in essentials.jl

function show(io::IO, x::Missing)
    print(io, "missing")
end

"""
    MissingException(msg)

Exception thrown when a [`missing`](@ref) value is encountered in a situation
where it is not supported. The error message, in the `msg` field
may provide more specific details.
"""
struct MissingException <: Exception
    msg::AbstractString
end

function showerror(io::IO, ex::MissingException)
    print(io, "MissingException: ", ex.msg)
end


function nonmissingtype(::Type{Union{T, Missing}}) where T
    T
end
function nonmissingtype(::Type{Missing})
    Union{}
end
function nonmissingtype(::Type{T}) where T
    T
end
function nonmissingtype(::Type{Any})
    Any
end

for U in (:Nothing, :Missing)
    @eval begin
        promote_rule(::Type{$U}, ::Type{T}) where {T} = Union{T, $U}
        promote_rule(::Type{Union{S,$U}}, ::Type{Any}) where {S} = Any
        promote_rule(::Type{Union{S,$U}}, ::Type{T}) where {T,S} = Union{promote_type(T, S), $U}
        promote_rule(::Type{Any}, ::Type{$U}) = Any
        promote_rule(::Type{$U}, ::Type{Any}) = Any
        # This definition is never actually used, but disambiguates the above definitions
        promote_rule(::Type{$U}, ::Type{$U}) = $U
    end
end
function promote_rule(::Type{Union{Nothing, Missing}}, ::Type{Any})
    Any
end
function promote_rule(::Type{Union{Nothing, Missing}}, ::Type{T}) where T
    Union{Nothing, Missing, T}
end
function promote_rule(::Type{Union{Nothing, Missing, S}}, ::Type{Any}) where S
    Any
end
function promote_rule(::Type{Union{Nothing, Missing, S}}, ::Type{T}) where {T, S}
    Union{Nothing, Missing, promote_type(T, S)}
end

function convert(::Type{Union{T, Missing}}, x::Union{T, Missing}) where T
    x
end
function convert(::Type{Union{T, Missing}}, x) where T
    convert(T, x)
end
# To fix ambiguities
function convert(::Type{Missing}, ::Missing)
    missing
end
function convert(::Type{Union{Nothing, Missing}}, x::Union{Nothing, Missing})
    x
end
function convert(::Type{Union{Nothing, Missing, T}}, x::Union{Nothing, Missing, T}) where T
    x
end
function convert(::Type{Union{Nothing, Missing}}, x)
    throw(MethodError(convert, (Union{Nothing, Missing}, x)))
end
# To print more appropriate message than "T not defined"
function convert(::Type{Missing}, x)
    throw(MethodError(convert, (Missing, x)))
end

# Comparison operators
function ==(::Missing, ::Missing)
    missing
end
function ==(::Missing, ::Any)
    missing
end
function ==(::Any, ::Missing)
    missing
end
# To fix ambiguity
function ==(::Missing, ::WeakRef)
    missing
end
function ==(::WeakRef, ::Missing)
    missing
end
function isequal(::Missing, ::Missing)
    true
end
function isequal(::Missing, ::Any)
    false
end
function isequal(::Any, ::Missing)
    false
end
function <(::Missing, ::Missing)
    missing
end
function <(::Missing, ::Any)
    missing
end
function <(::Any, ::Missing)
    missing
end
function isless(::Missing, ::Missing)
    false
end
function isless(::Missing, ::Any)
    false
end
function isless(::Any, ::Missing)
    true
end
function isapprox(::Missing, ::Missing; kwargs...)
    missing
end
function isapprox(::Missing, ::Any; kwargs...)
    missing
end
function isapprox(::Any, ::Missing; kwargs...)
    missing
end

# Unary operators/functions
for f in (:(!), :(~), :(+), :(-), :(zero), :(one), :(oneunit),
          :(isfinite), :(isinf), :(isodd),
          :(isinteger), :(isreal), :(isnan),
          :(iszero), :(transpose), :(adjoint), :(float), :(conj),
          :(abs), :(abs2), :(iseven), :(ispow2),
          :(real), :(imag), :(sign), :(inv))
    @eval ($f)(::Missing) = missing
end
for f in (:(Base.zero), :(Base.one), :(Base.oneunit))
    @eval ($f)(::Type{Missing}) = missing
    @eval function $(f)(::Type{Union{T, Missing}}) where T
        T === Any && throw(MethodError($f, (Any,)))  # To prevent StackOverflowError
        $f(T)
    end
end

# Binary operators/functions
for f in (:(+), :(-), :(*), :(/), :(^), :(div), :(mod), :(fld), :(rem))
    @eval begin
        # Scalar with missing
        ($f)(::Missing, ::Missing) = missing
        ($f)(::Missing, ::Number)  = missing
        ($f)(::Number,  ::Missing) = missing
    end
end

function min(::Missing, ::Missing)
    missing
end
function min(::Missing, ::Any)
    missing
end
function min(::Any, ::Missing)
    missing
end
function max(::Missing, ::Missing)
    missing
end
function max(::Missing, ::Any)
    missing
end
function max(::Any, ::Missing)
    missing
end

# Rounding and related functions
function round(::Missing, ::RoundingMode=RoundNearest; sigdigits::Integer=0, digits::Integer=0, base::Integer=0)
    missing
end
function round(::Type{>:Missing}, ::Missing, ::RoundingMode=RoundNearest)
    missing
end
function round(::Type{T}, ::Missing, ::RoundingMode=RoundNearest) where T
    throw(MissingException("cannot convert a missing value to type $(T): use Union{$(T), Missing} instead"))
end
function round(::Type{T}, x::Any, r::RoundingMode=RoundNearest) where $(Expr(:>:, :T, :Missing))
    round(nonmissingtype(T), x, r)
end
# to fix ambiguities
function round(::Type{T}, x::Rational, r::RoundingMode=RoundNearest) where $(Expr(:>:, :T, :Missing))
    round(nonmissingtype(T), x, r)
end
function round(::Type{T}, x::Rational{Bool}, r::RoundingMode=RoundNearest) where $(Expr(:>:, :T, :Missing))
    round(nonmissingtype(T), x, r)
end

# Handle ceil, floor, and trunc separately as they have no RoundingMode argument
for f in (:(ceil), :(floor), :(trunc))
    @eval begin
        ($f)(::Missing; sigdigits::Integer=0, digits::Integer=0, base::Integer=0) = missing
        ($f)(::Type{>:Missing}, ::Missing) = missing
        ($f)(::Type{T}, ::Missing) where {T} =
            throw(MissingException("cannot convert a missing value to type $T: use Union{$T, Missing} instead"))
        ($f)(::Type{T}, x::Any) where {T>:Missing} = $f(nonmissingtype(T), x)
        # to fix ambiguities
        ($f)(::Type{T}, x::Rational) where {T>:Missing} = $f(nonmissingtype(T), x)
    end
end

# to avoid ambiguity warnings
function ^(::Missing, ::Integer)
    missing
end

# Bit operators
function &(::Missing, ::Missing)
    missing
end
function &(a::Missing, b::Bool)
    ifelse(b, missing, false)
end
function &(b::Bool, a::Missing)
    ifelse(b, missing, false)
end
function &(::Missing, ::Integer)
    missing
end
function &(::Integer, ::Missing)
    missing
end
function |(::Missing, ::Missing)
    missing
end
function |(a::Missing, b::Bool)
    ifelse(b, true, missing)
end
function |(b::Bool, a::Missing)
    ifelse(b, true, missing)
end
function |(::Missing, ::Integer)
    missing
end
function |(::Integer, ::Missing)
    missing
end
function xor(::Missing, ::Missing)
    missing
end
function xor(a::Missing, b::Bool)
    missing
end
function xor(b::Bool, a::Missing)
    missing
end
function xor(::Missing, ::Integer)
    missing
end
function xor(::Integer, ::Missing)
    missing
end

function *(d::Missing, x::AbstractString)
    missing
end
function *(d::AbstractString, x::Missing)
    missing
end

function float(A::AbstractArray{Union{T, Missing}}) where {T}
    U = typeof(float(zero(T)))
    convert(AbstractArray{Union{U, Missing}}, A)
end
function float(A::AbstractArray{Missing})
    A
end

"""
    skipmissing(itr)

Return an iterator over the elements in `itr` skipping [`missing`](@ref) values.
The returned object can be indexed using indices of `itr` if the latter is indexable.
Indices corresponding to missing values are not valid: they are skipped by [`keys`](@ref)
and [`eachindex`](@ref), and a `MissingException` is thrown when trying to use them.

Use [`collect`](@ref) to obtain an `Array` containing the non-`missing` values in
`itr`. Note that even if `itr` is a multidimensional array, the result will always
be a `Vector` since it is not possible to remove missings while preserving dimensions
of the input.

# Examples
```jldoctest
julia> x = skipmissing([1, missing, 2])
Base.SkipMissing{Array{Union{Missing, Int64},1}}(Union{Missing, Int64}[1, missing, 2])

julia> sum(x)
3

julia> x[1]
1

julia> x[2]
ERROR: MissingException: the value at index (2,) is missing
[...]

julia> argmax(x)
3

julia> collect(keys(x))
2-element Array{Int64,1}:
 1
 3

julia> collect(skipmissing([1, missing, 2]))
2-element Array{Int64,1}:
 1
 2

julia> collect(skipmissing([1 missing; 2 missing]))
2-element Array{Int64,1}:
 1
 2
```
"""
skipmissing(itr) = SkipMissing(itr)

struct SkipMissing{T}
    x::T
end
function IteratorSize(::Type{<:SkipMissing})
    SizeUnknown()
end
function IteratorEltype(::Type{SkipMissing{T}}) where T
    IteratorEltype(T)
end
function eltype(::Type{SkipMissing{T}}) where T
    nonmissingtype(eltype(T))
end

function iterate(itr::SkipMissing, state...)
    y = iterate(itr.x, state...)
    y === nothing && return nothing
    item, state = y
    while item === missing
        y = iterate(itr.x, state)
        y === nothing && return nothing
        item, state = y
    end
    item, state
end

function IndexStyle(::Type{<:SkipMissing{T}}) where T
    IndexStyle(T)
end
function eachindex(itr::SkipMissing)
    Iterators.filter((i->begin
                @inbounds(itr.x[i]) !== missing
            end), eachindex(itr.x))
end
function keys(itr::SkipMissing)
    Iterators.filter((i->begin
                @inbounds(itr.x[i]) !== missing
            end), keys(itr.x))
end
@propagate_inbounds function getindex(itr::SkipMissing, I...)
    v = itr.x[I...]
    v === missing && throw(MissingException("the value at index $I is missing"))
    v
end

# Optimized mapreduce implementation
# The generic method is faster when !(eltype(A) >: Missing) since it does not need
# additional loops to identify the two first non-missing values of each block
function mapreduce(f, op, itr::SkipMissing{<:AbstractArray})
    _mapreduce(f, op, IndexStyle(itr.x), if $(Expr(:>:, :(eltype(itr.x)), :Missing))
            itr
        else
            itr.x
        end)
end

function _mapreduce(f, op, ::IndexLinear, itr::SkipMissing{<:AbstractArray})
    A = itr.x
    local ai
    inds = LinearIndices(A)
    i = first(inds)
    ilast = last(inds)
    while i <= ilast
        @inbounds ai = A[i]
        ai === missing || break
        i += 1
    end
    i > ilast && return mapreduce_empty(f, op, eltype(itr))
    a1 = ai
    i += 1
    while i <= ilast
        @inbounds ai = A[i]
        ai === missing || break
        i += 1
    end
    i > ilast && return mapreduce_first(f, op, a1)
    # We know A contains at least two non-missing entries: the result cannot be nothing
    something(mapreduce_impl(f, op, itr, first(inds), last(inds)))
end

function _mapreduce(f, op, ::IndexCartesian, itr::SkipMissing)
    mapfoldl(f, op, itr)
end

function mapreduce_impl(f, op, A::SkipMissing, ifirst::Integer, ilast::Integer)
    mapreduce_impl(f, op, A, ifirst, ilast, pairwise_blocksize(f, op))
end

# Returns nothing when the input contains only missing values, and Some(x) otherwise
@noinline function mapreduce_impl(f, op, itr::SkipMissing{<:AbstractArray},
                                  ifirst::Integer, ilast::Integer, blksize::Int)
    A = itr.x
    if ifirst == ilast
        @inbounds a1 = A[ifirst]
        if a1 === missing
            return nothing
        else
            return Some(mapreduce_first(f, op, a1))
        end
    elseif ifirst + blksize > ilast
        # sequential portion
        local ai
        i = ifirst
        while i <= ilast
            @inbounds ai = A[i]
            ai === missing || break
            i += 1
        end
        i > ilast && return nothing
        a1 = ai::eltype(itr)
        i += 1
        while i <= ilast
            @inbounds ai = A[i]
            ai === missing || break
            i += 1
        end
        i > ilast && return Some(mapreduce_first(f, op, a1))
        a2 = ai::eltype(itr)
        i += 1
        v = op(f(a1), f(a2))
        @simd for i = i:ilast
            @inbounds ai = A[i]
            if ai !== missing
                v = op(v, f(ai))
            end
        end
        return Some(v)
    else
        # pairwise portion
        imid = (ifirst + ilast) >> 1
        v1 = mapreduce_impl(f, op, itr, ifirst, imid, blksize)
        v2 = mapreduce_impl(f, op, itr, imid+1, ilast, blksize)
        if v1 === nothing && v2 === nothing
            return nothing
        elseif v1 === nothing
            return v2
        elseif v2 === nothing
            return v1
        else
            return Some(op(something(v1), something(v2)))
        end
    end
end

"""
    filter(f, itr::SkipMissing{<:AbstractArray})

Return a vector similar to the array wrapped by the given `SkipMissing` iterator
but with all missing elements and those for which `f` returns `false` removed.

!!! compat "Julia 1.2"
    This method requires Julia 1.2 or later.

# Examples
```jldoctest
julia> x = [1 2; missing 4]
2×2 Array{Union{Missing, Int64},2}:
 1         2
  missing  4

julia> filter(isodd, skipmissing(x))
1-element Array{Int64,1}:
 1
```
"""
function filter(f, itr::SkipMissing{<:AbstractArray})
    y = similar(itr.x, eltype(itr), 0)
    for xi in itr.x
        if xi !== missing && f(xi)
            push!(y, xi)
        end
    end
    y
end

"""
    coalesce(x, y...)

Return the first value in the arguments which is not equal to [`missing`](@ref),
if any. Otherwise return `missing`.

# Examples

```jldoctest
julia> coalesce(missing, 1)
1

julia> coalesce(1, missing)
1

julia> coalesce(nothing, 1)  # returns `nothing`

julia> coalesce(missing, missing)
missing
```
"""
function coalesce end

function coalesce()
    missing
end
function coalesce(x::Missing, y...)
    coalesce(y...)
end
function coalesce(x::Any, y...)
    x
end
