# This file is a part of Julia. License is MIT: https://julialang.org/license

# Document NTuple here where we have everything needed for the doc system
"""
    NTuple{N, T}

A compact way of representing the type for a tuple of length `N` where all elements are of type `T`.

# Examples
```jldoctest
julia> isa((1, 2, 3, 4, 5, 6), NTuple{6, Int})
true
```
"""
NTuple

## indexing ##

function length(@nospecialize(t::Tuple))
    nfields(t)
end
function firstindex(@nospecialize(t::Tuple))
    1
end
function lastindex(@nospecialize(t::Tuple))
    length(t)
end
function size(@nospecialize(t::Tuple), d::Integer)
    if d == 1
        length(t)
    else
        throw(ArgumentError("invalid tuple dimension $(d)"))
    end
end
function axes(@nospecialize(t::Tuple))
    (OneTo(length(t)),)
end
@eval getindex(@nospecialize(t::Tuple), i::Int) = getfield(t, i, $(Expr(:boundscheck)))
@eval getindex(@nospecialize(t::Tuple), i::Real) = getfield(t, convert(Int, i), $(Expr(:boundscheck)))
function getindex(t::Tuple, r::AbstractArray{<:Any, 1})
    ([t[ri] for ri = r]...,)
end
function getindex(t::Tuple, b::AbstractArray{Bool, 1})
    if length(b) == length(t)
        getindex(t, findall(b))
    else
        throw(BoundsError(t, b))
    end
end
function getindex(t::Tuple, c::Colon)
    t
end

# returns new tuple; N.B.: becomes no-op if i is out-of-bounds
function setindex(x::Tuple, v, i::Integer)
    @_inline_meta
    _setindex(v, i, x...)
end
function _setindex(v, i::Integer, first, tail...)
    @_inline_meta
    return (ifelse(i == 1, v, first), _setindex(v, i - 1, tail...)...)
end
function _setindex(v, i::Integer)
    ()
end


## iterating ##

function iterate(@nospecialize(t::Tuple), i::Int=1)
    @_inline_meta
    return (1 <= i <= length(t)) ? (@inbounds t[i], i + 1) : nothing
end

function keys(@nospecialize(t::Tuple))
    OneTo(length(t))
end

function prevind(@nospecialize(t::Tuple), i::Integer)
    Int(i) - 1
end
function nextind(@nospecialize(t::Tuple), i::Integer)
    Int(i) + 1
end

function keys(t::Tuple, t2::Tuple...)
    @_inline_meta
    OneTo(_maxlength(t, t2...))
end
function _maxlength(t::Tuple)
    length(t)
end
function _maxlength(t::Tuple, t2::Tuple, t3::Tuple...)
    @_inline_meta
    max(length(t), _maxlength(t2, t3...))
end

# this allows partial evaluation of bounded sequences of next() calls on tuples,
# while reducing to plain next() for arbitrary iterables.
function indexed_iterate(t::Tuple, i::Int, state=1)
    @_inline_meta
    (getfield(t, i), i + 1)
end
function indexed_iterate(a::Array, i::Int, state=1)
    @_inline_meta
    (a[i], i + 1)
end
function indexed_iterate(I, i)
    x = iterate(I)
    x === nothing && throw(BoundsError(I, i))
    x
end
function indexed_iterate(I, i, state)
    x = iterate(I, state)
    x === nothing && throw(BoundsError(I, i))
    x
end

# Use dispatch to avoid a branch in first
function first(::Tuple{})
    throw(ArgumentError("tuple must be non-empty"))
end
function first(t::Tuple)
    t[1]
end

# eltype

function eltype(::Type{Tuple{}})
    Bottom
end
function eltype(t::Type{<:Tuple{Vararg{E}}}) where {E}
    if @isdefined(E)
        return E
    else
        # TODO: need to guard against E being miscomputed by subtyping (ref #23017)
        # and compute the result manually in this case
        return _compute_eltype(t)
    end
end
function eltype(t::Type{<:Tuple})
    _compute_eltype(t)
end
function _compute_eltype(t::Type{<:Tuple})
    @_pure_meta
    t isa Union && return promote_typejoin(eltype(t.a), eltype(t.b))
    t´ = unwrap_unionall(t)
    r = Union{}
    for ti in t´.parameters
        r = promote_typejoin(r, rewrap_unionall(unwrapva(ti), t))
    end
    return r
end

# version of tail that doesn't throw on empty tuples (used in array indexing)
function safe_tail(t::Tuple)
    tail(t)
end
function safe_tail(t::Tuple{})
    ()
end

# front (the converse of tail: it skips the last entry)

"""
    front(x::Tuple)::Tuple

Return a `Tuple` consisting of all but the last component of `x`.

# Examples
```jldoctest
julia> Base.front((1,2,3))
(1, 2)

julia> Base.front(())
ERROR: ArgumentError: Cannot call front on an empty tuple.
```
"""
function front(t::Tuple)
    @_inline_meta
    _front(t...)
end
function _front()
    throw(ArgumentError("Cannot call front on an empty tuple."))
end
function _front(v)
    ()
end
function _front(v, t...)
    @_inline_meta
    (v, _front(t...)...)
end

## mapping ##

# 1 argument function
function map(f, t::Tuple{})
    ()
end
function map(f, t::Tuple{Any})
    (f(t[1]),)
end
function map(f, t::Tuple{Any, Any})
    (f(t[1]), f(t[2]))
end
function map(f, t::Tuple{Any, Any, Any})
    (f(t[1]), f(t[2]), f(t[3]))
end
function map(f, t::Tuple)
    @_inline_meta
    (f(t[1]), map(f, tail(t))...)
end
# stop inlining after some number of arguments to avoid code blowup
const Any16{N} = Tuple{Any,Any,Any,Any,Any,Any,Any,Any,
                       Any,Any,Any,Any,Any,Any,Any,Any,Vararg{Any,N}}
const All16{T,N} = Tuple{T,T,T,T,T,T,T,T,
                         T,T,T,T,T,T,T,T,Vararg{T,N}}
function map(f, t::Any16)
    n = length(t)
    A = Vector{Any}(undef, n)
    for i=1:n
        A[i] = f(t[i])
    end
    (A...,)
end
# 2 argument function
function map(f, t::Tuple{}, s::Tuple{})
    ()
end
function map(f, t::Tuple{Any}, s::Tuple{Any})
    (f(t[1], s[1]),)
end
function map(f, t::Tuple{Any, Any}, s::Tuple{Any, Any})
    (f(t[1], s[1]), f(t[2], s[2]))
end
function map(f, t::Tuple, s::Tuple)
    @_inline_meta
    (f(t[1],s[1]), map(f, tail(t), tail(s))...)
end
function map(f, t::Any16, s::Any16)
    n = length(t)
    A = Vector{Any}(undef, n)
    for i = 1:n
        A[i] = f(t[i], s[i])
    end
    (A...,)
end
# n argument function
function heads(ts::Tuple...)
    map((t->begin
                t[1]
            end), ts)
end
function tails(ts::Tuple...)
    map(tail, ts)
end
function map(f, ::Tuple{}...)
    ()
end
function map(f, t1::Tuple, t2::Tuple, ts::Tuple...)
    @_inline_meta
    (f(heads(t1, t2, ts...)...), map(f, tails(t1, t2, ts...)...)...)
end
function map(f, t1::Any16, t2::Any16, ts::Any16...)
    n = length(t1)
    A = Vector{Any}(undef, n)
    for i = 1:n
        A[i] = f(t1[i], t2[i], map(t -> t[i], ts)...)
    end
    (A...,)
end

# mapafoldl, based on afold in operators.jl
function mapafoldl(F, op, a)
    a
end
function mapafoldl(F, op, a, b)
    op(a, F(b))
end
function mapafoldl(F, op, a, b, c...)
    mapafoldl(F, op, op(a, F(b)), c...)
end
function mapafoldl(F,op,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,qs...)
    y = op(op(op(op(op(op(op(op(op(op(op(op(op(op(op(a,F(b)),F(c)),F(d)),F(e)),F(f)),F(g)),F(h)),F(i)),F(j)),F(k)),F(l)),F(m)),F(n)),F(o)),F(p))
    for x in qs; y = op(y,F(x)); end
    y
end
function mapfoldl_impl(f, op, nt::NamedTuple{(:init,)}, t::Tuple)
    mapafoldl(f, op, nt.init, t...)
end
function mapfoldl_impl(f, op, nt::NamedTuple{()}, t::Tuple)
    mapafoldl(f, op, f(t[1]), tail(t)...)
end
function mapfoldl_impl(f, op, nt::NamedTuple{()}, t::Tuple{})
    mapreduce_empty_iter(f, op, t, IteratorEltype(t))
end

# type-stable padding
function fill_to_length(t::NTuple{N, Any}, val, ::Val{N}) where N
    t
end
function fill_to_length(t::Tuple{}, val, ::Val{1})
    (val,)
end
function fill_to_length(t::Tuple{Any}, val, ::Val{2})
    (t..., val)
end
function fill_to_length(t::Tuple{}, val, ::Val{2})
    (val, val)
end
#function fill_to_length(t::Tuple, val, ::Val{N}) where {N}
#    @_inline_meta
#    return (t..., ntuple(i -> val, N - length(t))...)
#end

# constructing from an iterator

# only define these in Base, to avoid overwriting the constructors
# NOTE: this means this constructor must be avoided in Core.Compiler!
if nameof(@__MODULE__) === :Base

(::Type{T})(x::Tuple) where {T<:Tuple} = convert(T, x)  # still use `convert` for tuples

(::Type{T})(itr) where {T<:Tuple} = _totuple(T, itr)

_totuple(::Type{Tuple{}}, itr, s...) = ()

function _totuple_err(@nospecialize T)
    @_noinline_meta
    throw(ArgumentError("too few elements for tuple type $T"))
end

function _totuple(T, itr, s...)
    @_inline_meta
    y = iterate(itr, s...)
    y === nothing && _totuple_err(T)
    (convert(tuple_type_head(T), y[1]), _totuple(tuple_type_tail(T), itr, y[2])...)
end

# use iterative algorithm for long tuples
function _totuple(T::Type{All16{E,N}}, itr) where {E,N}
    len = N+16
    elts = collect(E, Iterators.take(itr,len))
    if length(elts) != len
        _totuple_err(T)
    end
    (elts...,)
end

_totuple(::Type{Tuple{Vararg{E}}}, itr, s...) where {E} = (collect(E, Iterators.rest(itr,s...))...,)

_totuple(::Type{Tuple}, itr, s...) = (collect(Iterators.rest(itr,s...))...,)

end

## comparison ##

function isequal(t1::Tuple, t2::Tuple)
    length(t1) == length(t2) && _isequal(t1, t2)
end
function _isequal(t1::Tuple{}, t2::Tuple{})
    true
end
function _isequal(t1::Tuple{Any}, t2::Tuple{Any})
    isequal(t1[1], t2[1])
end
function _isequal(t1::Tuple, t2::Tuple)
    isequal(t1[1], t2[1]) && _isequal(tail(t1), tail(t2))
end
function _isequal(t1::Any16, t2::Any16)
    for i = 1:length(t1)
        if !isequal(t1[i], t2[i])
            return false
        end
    end
    return true
end

function ==(t1::Tuple, t2::Tuple)
    length(t1) == length(t2) && _eq(t1, t2)
end
function _eq(t1::Tuple{}, t2::Tuple{})
    true
end
function _eq_missing(t1::Tuple{}, t2::Tuple{})
    missing
end
function _eq(t1::Tuple, t2::Tuple)
    eq = t1[1] == t2[1]
    if eq === false
        return false
    elseif ismissing(eq)
        return _eq_missing(tail(t1), tail(t2))
    else
        return _eq(tail(t1), tail(t2))
    end
end
function _eq_missing(t1::Tuple, t2::Tuple)
    eq = t1[1] == t2[1]
    if eq === false
        return false
    else
        return _eq_missing(tail(t1), tail(t2))
    end
end
function _eq(t1::Any16, t2::Any16)
    anymissing = false
    for i = 1:length(t1)
        eq = (t1[i] == t2[i])
        if ismissing(eq)
            anymissing = true
        elseif !eq
           return false
       end
    end
    return anymissing ? missing : true
end

const tuplehash_seed = UInt === UInt64 ? 0x77cfa1eef01bca90 : 0xf01bca90
function hash(::Tuple{}, h::UInt)
    h + tuplehash_seed
end
function hash(t::Tuple, h::UInt)
    hash(t[1], hash(tail(t), h))
end
function hash(t::Any16, h::UInt)
    out = h + tuplehash_seed
    for i = length(t):-1:1
        out = hash(t[i], out)
    end
    return out
end

function <(::Tuple{}, ::Tuple{})
    false
end
function <(::Tuple{}, ::Tuple)
    true
end
function <(::Tuple, ::Tuple{})
    false
end
function <(t1::Tuple, t2::Tuple)
    a, b = t1[1], t2[1]
    eq = (a == b)
    if ismissing(eq)
        return missing
    elseif !eq
        return a < b
    end
    return tail(t1) < tail(t2)
end
function <(t1::Any16, t2::Any16)
    n1, n2 = length(t1), length(t2)
    for i = 1:min(n1, n2)
        a, b = t1[i], t2[i]
        eq = (a == b)
        if ismissing(eq)
            return missing
        elseif !eq
           return a < b
        end
    end
    return n1 < n2
end

function isless(::Tuple{}, ::Tuple{})
    false
end
function isless(::Tuple{}, ::Tuple)
    true
end
function isless(::Tuple, ::Tuple{})
    false
end

"""
    isless(t1::Tuple, t2::Tuple)

Returns true when t1 is less than t2 in lexicographic order.
"""
function isless(t1::Tuple, t2::Tuple)
    a, b = t1[1], t2[1]
    isless(a, b) || (isequal(a, b) && isless(tail(t1), tail(t2)))
end
function isless(t1::Any16, t2::Any16)
    n1, n2 = length(t1), length(t2)
    for i = 1:min(n1, n2)
        a, b = t1[i], t2[i]
        if !isequal(a, b)
            return isless(a, b)
        end
    end
    return n1 < n2
end

## functions ##

function isempty(x::Tuple{})
    true
end
function isempty(@nospecialize(x::Tuple))
    false
end

function revargs()
    ()
end
function revargs(x, r...)
    (revargs(r...)..., x)
end

function reverse(t::Tuple)
    revargs(t...)
end

## specialized reduction ##

# TODO: these definitions cannot yet be combined, since +(x...)
# where x might be any tuple matches too many methods.
# TODO: this is inconsistent with the regular sum in cases where the arguments
# require size promotion to system size.
function sum(x::Tuple{Any, Vararg{Any}})
    +(x...)
end

# NOTE: should remove, but often used on array sizes
# TODO: this is inconsistent with the regular prod in cases where the arguments
# require size promotion to system size.
function prod(x::Tuple{})
    1
end
function prod(x::Tuple{Any, Vararg{Any}})
    (*)(x...)
end

function all(x::Tuple{})
    true
end
function all(x::Tuple{Bool})
    x[1]
end
function all(x::Tuple{Bool, Bool})
    x[1] & x[2]
end
function all(x::Tuple{Bool, Bool, Bool})
    (x[1] & x[2]) & x[3]
end
# use generic reductions for the rest

function any(x::Tuple{})
    false
end
function any(x::Tuple{Bool})
    x[1]
end
function any(x::Tuple{Bool, Bool})
    x[1] | x[2]
end
function any(x::Tuple{Bool, Bool, Bool})
    (x[1] | x[2]) | x[3]
end

# equivalent to any(f, t), to be used only in bootstrap
function _tuple_any(f::Function, t::Tuple)
    _tuple_any(f, false, t...)
end
function _tuple_any(f::Function, tf::Bool, a, b...)
    @_inline_meta
    _tuple_any(f, tf | f(a), b...)
end
function _tuple_any(f::Function, tf::Bool)
    tf
end

"""
    empty(x::Tuple)

Returns an empty tuple, `()`.
"""
empty(@nospecialize x::Tuple) = ()
