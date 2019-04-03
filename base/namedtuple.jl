# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    NamedTuple

`NamedTuple`s are, as their name suggests, named [`Tuple`](@ref)s. That is, they're a
tuple-like collection of values, where each entry has a unique name, represented as a
[`Symbol`](@ref). Like `Tuple`s, `NamedTuple`s are immutable; neither the names nor the values
can be modified in place after construction.

Accessing the value associated with a name in a named tuple can be done using field
access syntax, e.g. `x.a`, or using [`getindex`](@ref), e.g. `x[:a]`. A tuple of the
names can be obtained using [`keys`](@ref), and a tuple of the values can be obtained
using [`values`](@ref).

!!! note
    Iteration over `NamedTuple`s produces the *values* without the names. (See example
    below.) To iterate over the name-value pairs, use the [`pairs`](@ref) function.

# Examples
```jldoctest
julia> x = (a=1, b=2)
(a = 1, b = 2)

julia> x.a
1

julia> x[:a]
1

julia> keys(x)
(:a, :b)

julia> values(x)
(1, 2)

julia> collect(x)
2-element Array{Int64,1}:
 1
 2

julia> collect(pairs(x))
2-element Array{Pair{Symbol,Int64},1}:
 :a => 1
 :b => 2
```
"""
Core.NamedTuple

if nameof(@__MODULE__) === :Base

"""
    NamedTuple{names,T}(args::Tuple)

Construct a named tuple with the given `names` (a tuple of Symbols) and field types `T`
(a `Tuple` type) from a tuple of values.
"""
function NamedTuple{names,T}(args::Tuple) where {names, T <: Tuple}
    if length(args) != length(names)
        throw(ArgumentError("Wrong number of arguments to named tuple constructor."))
    end
    NamedTuple{names,T}(T(args))
end

"""
    NamedTuple{names}(nt::NamedTuple)

Construct a named tuple by selecting fields in `names` (a tuple of Symbols) from
another named tuple.
"""
function NamedTuple{names}(nt::NamedTuple) where {names}
    if @generated
        types = Tuple{(fieldtype(nt, n) for n in names)...}
        Expr(:new, :(NamedTuple{names, $types}), Any[ :(getfield(nt, $(QuoteNode(n)))) for n in names ]...)
    else
        types = Tuple{(fieldtype(typeof(nt), n) for n in names)...}
        NamedTuple{names, types}(Tuple(getfield(nt, n) for n in names))
    end
end

NamedTuple{names, T}(itr) where {names, T <: Tuple} = NamedTuple{names, T}(T(itr))
NamedTuple{names}(itr) where {names} = NamedTuple{names}(Tuple(itr))

end # if Base

function length(t::NamedTuple)
    nfields(t)
end
function iterate(t::NamedTuple, iter=1)
    if iter > nfields(t)
        nothing
    else
        (getfield(t, iter), iter + 1)
    end
end
function firstindex(t::NamedTuple)
    1
end
function lastindex(t::NamedTuple)
    nfields(t)
end
function getindex(t::NamedTuple, i::Int)
    getfield(t, i)
end
function getindex(t::NamedTuple, i::Symbol)
    getfield(t, i)
end
function indexed_iterate(t::NamedTuple, i::Int, state=1)
    (getfield(t, i), i + 1)
end
function isempty(::NamedTuple{()})
    true
end
function isempty(::NamedTuple)
    false
end

function convert(::Type{NamedTuple{names, T}}, nt::NamedTuple{names, T}) where {names, T <: Tuple}
    nt
end
function convert(::Type{NamedTuple{names}}, nt::NamedTuple{names}) where names
    nt
end

function convert(::Type{NamedTuple{names,T}}, nt::NamedTuple{names}) where {names,T<:Tuple}
    NamedTuple{names,T}(T(nt))
end

if nameof(@__MODULE__) === :Base
    Tuple(nt::NamedTuple) = (nt...,)
    (::Type{T})(nt::NamedTuple) where {T <: Tuple} = convert(T, Tuple(nt))
end

function show(io::IO, t::NamedTuple)
    n = nfields(t)
    for i = 1:n
        # if field types aren't concrete, show full type
        if typeof(getfield(t, i)) !== fieldtype(typeof(t), i)
            show(io, typeof(t))
            print(io, "(")
            show(io, Tuple(t))
            print(io, ")")
            return
        end
    end
    if n == 0
        print(io, "NamedTuple()")
    else
        typeinfo = get(io, :typeinfo, Any)
        print(io, "(")
        for i = 1:n
            print(io, fieldname(typeof(t),i), " = ")
            show(IOContext(io, :typeinfo =>
                           t isa typeinfo <: NamedTuple ? fieldtype(typeinfo, i) : Any),
                 getfield(t, i))
            if n == 1
                print(io, ",")
            elseif i < n
                print(io, ", ")
            end
        end
        print(io, ")")
    end
end

function eltype(::Type{NamedTuple{names, T}}) where {names, T}
    eltype(T)
end

function (a::NamedTuple{n} == b::NamedTuple{n}) where n
    Tuple(a) == Tuple(b)
end
function ==(a::NamedTuple, b::NamedTuple)
    false
end

function isequal(a::NamedTuple{n}, b::NamedTuple{n}) where n
    isequal(Tuple(a), Tuple(b))
end
function isequal(a::NamedTuple, b::NamedTuple)
    false
end

function _nt_names(::NamedTuple{names}) where names
    names
end
function _nt_names(::Type{T}) where {names, T <: NamedTuple{names}}
    names
end

function hash(x::NamedTuple, h::UInt)
    xor(objectid(_nt_names(x)), hash(Tuple(x), h))
end

function isless(a::NamedTuple{n}, b::NamedTuple{n}) where n
    isless(Tuple(a), Tuple(b))
end
# TODO: case where one argument's names are a prefix of the other's

function same_names(::NamedTuple{names}...) where names
    true
end
function same_names(::NamedTuple...)
    false
end

# NOTE: this method signature makes sure we don't define map(f)
function map(f, nt::NamedTuple{names}, nts::NamedTuple...) where names
    if !same_names(nt, nts...)
        throw(ArgumentError("Named tuple names do not match."))
    end
    NamedTuple{names}(map(f, map(Tuple, (nt, nts...))...))
end

# a version of `in` for the older world these generated functions run in
@pure function sym_in(x::Symbol, itr::Tuple{Vararg{Symbol}})
    for y in itr
        y === x && return true
    end
    return false
end

@pure function merge_names(an::Tuple{Vararg{Symbol}}, bn::Tuple{Vararg{Symbol}})
    names = Symbol[an...]
    for n in bn
        if !sym_in(n, an)
            push!(names, n)
        end
    end
    (names...,)
end

@pure function merge_types(names::Tuple{Vararg{Symbol}}, a::Type{<:NamedTuple}, b::Type{<:NamedTuple})
    bn = _nt_names(b)
    Tuple{Any[ fieldtype(sym_in(n, bn) ? b : a, n) for n in names ]...}
end

"""
    merge(a::NamedTuple, bs::NamedTuple...)

Construct a new named tuple by merging two or more existing ones, in a left-associative
manner. Merging proceeds left-to-right, between pairs of named tuples, and so the order of fields
present in both the leftmost and rightmost named tuples take the same position as they are found in the
leftmost named tuple. However, values are taken from matching fields in the rightmost named tuple that
contains that field. Fields present in only the rightmost named tuple of a pair are appended at the end.
A fallback is implemented for when only a single named tuple is supplied,
with signature `merge(a::NamedTuple)`.

!!! compat "Julia 1.1"
    Merging 3 or more `NamedTuple` requires at least Julia 1.1.

# Examples
```jldoctest
julia> merge((a=1, b=2, c=3), (b=4, d=5))
(a = 1, b = 4, c = 3, d = 5)
```

```jldoctest
julia> merge((a=1, b=2), (b=3, c=(d=1,)), (c=(d=2,),))
(a = 1, b = 3, c = (d = 2,))
```
"""
function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    if @generated
        names = merge_names(an, bn)
        types = merge_types(names, a, b)
        vals = Any[ :(getfield($(sym_in(n, bn) ? :b : :a), $(QuoteNode(n)))) for n in names ]
        :( NamedTuple{$names,$types}(($(vals...),)) )
    else
        names = merge_names(an, bn)
        types = merge_types(names, typeof(a), typeof(b))
        NamedTuple{names,types}(map(n->getfield(sym_in(n, bn) ? b : a, n), names))
    end
end

function merge(a::NamedTuple{()}, b::NamedTuple)
    b
end

function merge(a::NamedTuple, b::Iterators.Pairs{<:Any, <:Any, <:Any, <:NamedTuple})
    merge(a, b.data)
end

function merge(a::NamedTuple, b::NamedTuple, cs::NamedTuple...)
    merge(merge(a, b), cs...)
end

function merge(a::NamedTuple)
    a
end

"""
    merge(a::NamedTuple, iterable)

Interpret an iterable of key-value pairs as a named tuple, and perform a merge.

```jldoctest
julia> merge((a=1, b=2, c=3), [:b=>4, :d=>5])
(a = 1, b = 4, c = 3, d = 5)
```
"""
function merge(a::NamedTuple, itr)
    names = Symbol[]
    vals = Any[]
    inds = IdDict()
    for (k,v) in itr
        oldind = get(inds, k, 0)
        if oldind > 0
            vals[oldind] = v
        else
            push!(names, k)
            push!(vals, v)
            inds[k] = length(names)
        end
    end
    merge(a, NamedTuple{(names...,)}((vals...,)))
end

function keys(nt::NamedTuple{names}) where names
    names
end
function values(nt::NamedTuple)
    Tuple(nt)
end
function haskey(nt::NamedTuple, key::Union{Integer, Symbol})
    isdefined(nt, key)
end
function get(nt::NamedTuple, key::Union{Integer, Symbol}, default)
    if haskey(nt, key)
        getfield(nt, key)
    else
        default
    end
end
function get(f::Callable, nt::NamedTuple, key::Union{Integer, Symbol})
    if haskey(nt, key)
        getfield(nt, key)
    else
        f()
    end
end
function tail(t::NamedTuple{names}) where names
    NamedTuple{tail(names)}(t)
end

@pure function diff_names(an::Tuple{Vararg{Symbol}}, bn::Tuple{Vararg{Symbol}})
    names = Symbol[]
    for n in an
        if !sym_in(n, bn)
            push!(names, n)
        end
    end
    (names...,)
end

"""
    structdiff(a::NamedTuple{an}, b::Union{NamedTuple{bn},Type{NamedTuple{bn}}}) where {an,bn}

Construct a copy of named tuple `a`, except with fields that exist in `b` removed.
`b` can be a named tuple, or a type of the form `NamedTuple{field_names}`.
"""
function structdiff(a::NamedTuple{an}, b::Union{NamedTuple{bn}, Type{NamedTuple{bn}}}) where {an, bn}
    if @generated
        names = diff_names(an, bn)
        types = Tuple{Any[ fieldtype(a, n) for n in names ]...}
        vals = Any[ :(getfield(a, $(QuoteNode(n)))) for n in names ]
        :( NamedTuple{$names,$types}(($(vals...),)) )
    else
        names = diff_names(an, bn)
        types = Tuple{Any[ fieldtype(typeof(a), n) for n in names ]...}
        NamedTuple{names,types}(map(n->getfield(a, n), names))
    end
end
