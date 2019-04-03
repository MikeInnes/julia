# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
Methods for working with Iterators.
"""
module Iterators

# small dance to make this work from Base or Intrinsics
import ..@__MODULE__, ..parentmodule
const Base = parentmodule(@__MODULE__)
using .Base:
    @inline, Pair, AbstractDict, IndexLinear, IndexCartesian, IndexStyle, AbstractVector, Vector,
    tail, tuple_type_head, tuple_type_tail, tuple_type_cons, SizeUnknown, HasLength, HasShape,
    IsInfinite, EltypeUnknown, HasEltype, OneTo, @propagate_inbounds, Generator, AbstractRange,
    LinearIndices, (:), |, +, -, !==, !, <=, <, missing, map, any

import .Base:
    first, last,
    isempty, length, size, axes, ndims,
    eltype, IteratorSize, IteratorEltype,
    haskey, keys, values, pairs,
    getindex, setindex!, get, iterate,
    popfirst!, isdone, peek

export enumerate, zip, rest, countfrom, take, drop, cycle, repeated, product, flatten, partition

function tail_if_any(::Tuple{})
    ()
end
function tail_if_any(x::Tuple)
    tail(x)
end

function _min_length(a, b, ::IsInfinite, ::IsInfinite)
    min(length(a), length(b))
end # inherit behaviour, error
function _min_length(a, b, A, ::IsInfinite)
    length(a)
end
function _min_length(a, b, ::IsInfinite, B)
    length(b)
end
function _min_length(a, b, A, B)
    min(length(a), length(b))
end

function _diff_length(a, b, A, ::IsInfinite)
    0
end
function _diff_length(a, b, ::IsInfinite, ::IsInfinite)
    0
end
function _diff_length(a, b, ::IsInfinite, B)
    length(a)
end # inherit behaviour, error
function _diff_length(a, b, A, B)
    max(length(a) - length(b), 0)
end

function and_iteratorsize(isz::T, ::T) where T
    isz
end
function and_iteratorsize(::HasLength, ::HasShape)
    HasLength()
end
function and_iteratorsize(::HasShape, ::HasLength)
    HasLength()
end
function and_iteratorsize(a, b)
    SizeUnknown()
end

function and_iteratoreltype(iel::T, ::T) where T
    iel
end
function and_iteratoreltype(a, b)
    EltypeUnknown()
end

## Reverse-order iteration for arrays and other collections.  Collections
## should implement iterate etcetera if possible/practical.
"""
    Iterators.reverse(itr)

Given an iterator `itr`, then `reverse(itr)` is an iterator over the
same collection but in the reverse order.

This iterator is "lazy" in that it does not make a copy of the collection in
order to reverse it; see [`Base.reverse`](@ref) for an eager implementation.

Not all iterator types `T` support reverse-order iteration.  If `T`
doesn't, then iterating over `Iterators.reverse(itr::T)` will throw a [`MethodError`](@ref)
because of the missing [`iterate`](@ref) methods for `Iterators.Reverse{T}`.
(To implement these methods, the original iterator
`itr::T` can be obtained from `r = Iterators.reverse(itr)` by `r.itr`.)

# Examples
```jldoctest
julia> foreach(println, Iterators.reverse(1:5))
5
4
3
2
1
```
"""
reverse(itr) = Reverse(itr)

struct Reverse{T}
    itr::T
end
function eltype(::Type{Reverse{T}}) where T
    eltype(T)
end
function length(r::Reverse)
    length(r.itr)
end
function size(r::Reverse)
    size(r.itr)
end
function IteratorSize(::Type{Reverse{T}}) where T
    IteratorSize(T)
end
function IteratorEltype(::Type{Reverse{T}}) where T
    IteratorEltype(T)
end
function last(r::Reverse)
    first(r.itr)
end # the first shall be last
function first(r::Reverse)
    last(r.itr)
end # and the last shall be first

# reverse-order array iterators: assumes more-specialized Reverse for eachindex
@propagate_inbounds function iterate(A::Reverse{<:AbstractArray}, state=(reverse(eachindex(A.itr)),))
    y = iterate(state...)
    y === nothing && return y
    idx, itrs = y
    (A.itr[idx], (state[1], itrs))
end

function reverse(R::AbstractRange)
    Base.reverse(R)
end # copying ranges is cheap
function reverse(G::Generator)
    Generator(G.f, reverse(G.iter))
end
function reverse(r::Reverse)
    r.itr
end
function reverse(x::Union{Number, AbstractChar})
    x
end
function reverse(p::Pair)
    Base.reverse(p)
end # copying pairs is cheap

function iterate(r::Reverse{<:Tuple}, i::Int=length(r.itr))
    if i < 1
        nothing
    else
        (r.itr[i], i - 1)
    end
end

# enumerate

struct Enumerate{I}
    itr::I
end

"""
    enumerate(iter)

An iterator that yields `(i, x)` where `i` is a counter starting at 1,
and `x` is the `i`th value from the given iterator. It's useful when
you need not only the values `x` over which you are iterating, but
also the number of iterations so far. Note that `i` may not be valid
for indexing `iter`; it's also possible that `x != iter[i]`, if `iter`
has indices that do not start at 1. See the `pairs(IndexLinear(),
iter)` method if you want to ensure that `i` is an index.

# Examples
```jldoctest
julia> a = ["a", "b", "c"];

julia> for (index, value) in enumerate(a)
           println("\$index \$value")
       end
1 a
2 b
3 c
```
"""
enumerate(iter) = Enumerate(iter)

function length(e::Enumerate)
    length(e.itr)
end
function size(e::Enumerate)
    size(e.itr)
end
@propagate_inbounds function iterate(e::Enumerate, state=(1,))
    i, rest = state[1], tail(state)
    n = iterate(e.itr, rest...)
    n === nothing && return n
    (i, n[1]), (i+1, n[2])
end

function eltype(::Type{Enumerate{I}}) where I
    Tuple{Int, eltype(I)}
end

function IteratorSize(::Type{Enumerate{I}}) where I
    IteratorSize(I)
end
function IteratorEltype(::Type{Enumerate{I}}) where I
    IteratorEltype(I)
end

@inline function iterate(r::Reverse{<:Enumerate})
    ri = reverse(r.itr.itr)
    iterate(r, (length(ri), ri))
end
@inline function iterate(r::Reverse{<:Enumerate}, state)
    i, ri, rest = state[1], state[2], tail(tail(state))
    n = iterate(ri, rest...)
    n === nothing && return n
    (i, n[1]), (i-1, ri, n[2])
end

"""
    Iterators.Pairs(values, keys) <: AbstractDict{eltype(keys), eltype(values)}

Transforms an indexable container into an Dictionary-view of the same data.
Modifying the key-space of the underlying data may invalidate this object.
"""
struct Pairs{K, V, I, A} <: AbstractDict{K, V}
    data::A
    itr::I
    Pairs(data::A, itr::I) where {A, I} = new{eltype(I), eltype(A), I, A}(data, itr)
end

"""
    pairs(IndexLinear(), A)
    pairs(IndexCartesian(), A)
    pairs(IndexStyle(A), A)

An iterator that accesses each element of the array `A`, returning
`i => x`, where `i` is the index for the element and `x = A[i]`.
Identical to `pairs(A)`, except that the style of index can be selected.
Also similar to `enumerate(A)`, except `i` will be a valid index
for `A`, while `enumerate` always counts from 1 regardless of the indices
of `A`.

Specifying [`IndexLinear()`](@ref) ensures that `i` will be an integer;
specifying [`IndexCartesian()`](@ref) ensures that `i` will be a
[`CartesianIndex`](@ref); specifying `IndexStyle(A)` chooses whichever has
been defined as the native indexing style for array `A`.

Mutation of the bounds of the underlying array will invalidate this iterator.

# Examples
```jldoctest
julia> A = ["a" "d"; "b" "e"; "c" "f"];

julia> for (index, value) in pairs(IndexStyle(A), A)
           println("\$index \$value")
       end
1 a
2 b
3 c
4 d
5 e
6 f

julia> S = view(A, 1:2, :);

julia> for (index, value) in pairs(IndexStyle(S), S)
           println("\$index \$value")
       end
CartesianIndex(1, 1) a
CartesianIndex(2, 1) b
CartesianIndex(1, 2) d
CartesianIndex(2, 2) e
```

See also: [`IndexStyle`](@ref), [`axes`](@ref).
"""
pairs(::IndexLinear,    A::AbstractArray) = Pairs(A, LinearIndices(A))
function pairs(::IndexCartesian, A::AbstractArray)
    Pairs(A, CartesianIndices(axes(A)))
end

# preserve indexing capabilities for known indexable types
# faster than zip(keys(a), values(a)) for arrays
function pairs(A::AbstractArray)
    pairs(IndexCartesian(), A)
end
function pairs(A::AbstractVector)
    pairs(IndexLinear(), A)
end
function pairs(tuple::Tuple)
    Pairs(tuple, keys(tuple))
end
function pairs(nt::NamedTuple)
    Pairs(nt, keys(nt))
end
function pairs(v::Core.SimpleVector)
    Pairs(v, LinearIndices(v))
end
# pairs(v::Pairs) = v # listed for reference, but already defined from being an AbstractDict

function length(v::Pairs)
    length(v.itr)
end
function axes(v::Pairs)
    axes(v.itr)
end
function size(v::Pairs)
    size(v.itr)
end
@propagate_inbounds function iterate(v::Pairs{K, V}, state...) where {K, V}
    x = iterate(v.itr, state...)
    x === nothing && return x
    indx, n = x
    item = v.data[indx]
    return (Pair{K, V}(indx, item), n)
end
@inline isdone(v::Pairs, state...) = isdone(v.itr, state...)

function IteratorSize(::Type{<:Pairs{<:Any, <:Any, I}}) where I
    IteratorSize(I)
end
function IteratorSize(::Type{<:Pairs{<:Any, <:Any, <:Base.AbstractUnitRange, <:Tuple}})
    HasLength()
end

function reverse(v::Pairs)
    Pairs(v.data, reverse(v.itr))
end

function haskey(v::Pairs, key)
    key in v.itr
end
function keys(v::Pairs)
    v.itr
end
function values(v::Pairs)
    v.data
end
function getindex(v::Pairs, key)
    v.data[key]
end
function setindex!(v::Pairs, value, key)
    v.data[key] = value
    v
end
function get(v::Pairs, key, default)
    get(v.data, key, default)
end
function get(f::Base.Callable, v::Pairs, key)
    get(f, v.data, key)
end

# zip

function zip_iteratorsize(a, b)
    and_iteratorsize(a, b)
end # as `and_iteratorsize` but inherit `Union{HasLength,IsInfinite}` of the shorter iterator
function zip_iteratorsize(::HasLength, ::IsInfinite)
    HasLength()
end
function zip_iteratorsize(::HasShape, ::IsInfinite)
    HasLength()
end
function zip_iteratorsize(a::IsInfinite, b)
    zip_iteratorsize(b, a)
end
function zip_iteratorsize(a::IsInfinite, b::IsInfinite)
    IsInfinite()
end

struct Zip{Is<:Tuple}
    is::Is
end

"""
    zip(iters...)

Run multiple iterators at the same time, until any of them is exhausted. The value type of
the `zip` iterator is a tuple of values of its subiterators.

Note: `zip` orders the calls to its subiterators in such a way that stateful iterators will
not advance when another iterator finishes in the current iteration.

# Examples
```jldoctest
julia> a = 1:5
1:5

julia> b = ["e","d","b","c","a"]
5-element Array{String,1}:
 "e"
 "d"
 "b"
 "c"
 "a"

julia> c = zip(a,b)
Base.Iterators.Zip{Tuple{UnitRange{Int64},Array{String,1}}}((1:5, ["e", "d", "b", "c", "a"]))

julia> length(c)
5

julia> first(c)
(1, "e")
```
"""
zip(a...) = Zip(a)
function length(z::Zip)
    n = _zip_min_length(z.is)
    n === nothing && throw(ArgumentError("iterator is of undefined length"))
    return n
end
function _zip_min_length(is)
    i = is[1]
    n = _zip_min_length(tail(is))
    if IteratorSize(i) isa IsInfinite
        return n
    else
        return n === nothing ? length(i) : min(n, length(i))
    end
end
function _zip_min_length(is::Tuple{})
    nothing
end
function size(z::Zip)
    _promote_shape(map(size, z.is)...)
end
function axes(z::Zip)
    _promote_shape(map(axes, z.is)...)
end
function _promote_shape(a, b...)
    promote_shape(a, _promote_shape(b...))
end
function _promote_shape(a)
    a
end
function eltype(::Type{Zip{Is}}) where Is <: Tuple
    _zip_eltype(Is)
end
function _zip_eltype(::Type{Is}) where Is <: Tuple
    tuple_type_cons(eltype(tuple_type_head(Is)), _zip_eltype(tuple_type_tail(Is)))
end
function _zip_eltype(::Type{Tuple{}})
    Tuple{}
end
@inline isdone(z::Zip) = _zip_any_isdone(z.is, map(_ -> (), z.is))
@inline isdone(z::Zip, ss) = _zip_any_isdone(z.is, map(tuple, ss))
@inline function _zip_any_isdone(is, ss)
    d1 = isdone(is[1], ss[1]...)
    d1 === true && return true
    return d1 | _zip_any_isdone(tail(is), tail(ss))
end
@inline _zip_any_isdone(::Tuple{}, ::Tuple{}) = false

@propagate_inbounds iterate(z::Zip) = _zip_iterate_all(z.is, map(_ -> (), z.is))
@propagate_inbounds iterate(z::Zip, ss) = _zip_iterate_all(z.is, map(tuple, ss))

# This first queries isdone from every iterator. If any gives true, it immediately returns
# nothing. It then iterates all those where isdone returned missing, afterwards all those
# it returned false, again terminating immediately if any iterator is exhausted. Finally,
# the results are interleaved appropriately.
@propagate_inbounds function _zip_iterate_all(is, ss)
    d, ds = _zip_isdone(is, ss)
    d && return nothing
    xs1 = _zip_iterate_some(is, ss, ds, missing)
    xs1 === nothing && return nothing
    xs2 = _zip_iterate_some(is, ss, ds, false)
    xs2 === nothing && return nothing
    return _zip_iterate_interleave(xs1, xs2, ds)
end

@propagate_inbounds function _zip_iterate_some(is, ss, ds::Tuple{T,Vararg{Any}}, f::T) where T
    x = iterate(is[1], ss[1]...)
    x === nothing && return nothing
    y = _zip_iterate_some(tail(is), tail(ss), tail(ds), f)
    y === nothing && return nothing
    return (x, y...)
end
@propagate_inbounds _zip_iterate_some(is, ss, ds::Tuple{Any,Vararg{Any}}, f) =
    _zip_iterate_some(tail(is), tail(ss), tail(ds), f)
function _zip_iterate_some(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Any)
    ()
end

function _zip_iterate_interleave(xs1, xs2, ds)
    t = _zip_iterate_interleave(tail(xs1), xs2, tail(ds))
    ((xs1[1][1], t[1]...), (xs1[1][2], t[2]...))
end
function _zip_iterate_interleave(xs1, xs2, ds::Tuple{Bool,Vararg{Any}})
    t = _zip_iterate_interleave(xs1, tail(xs2), tail(ds))
    ((xs2[1][1], t[1]...), (xs2[1][2], t[2]...))
end
function _zip_iterate_interleave(::Tuple{}, ::Tuple{}, ::Tuple{})
    ((), ())
end

function _zip_isdone(is, ss)
    d = isdone(is[1], ss[1]...)
    d´, ds = _zip_isdone(tail(is), tail(ss))
    return (d === true || d´, (d, ds...))
end
function _zip_isdone(::Tuple{}, ::Tuple{})
    (false, ())
end

function IteratorSize(::Type{Zip{Is}}) where Is <: Tuple
    _zip_iterator_size(Is)
end
function _zip_iterator_size(::Type{Is}) where Is <: Tuple
    zip_iteratorsize(IteratorSize(tuple_type_head(Is)), _zip_iterator_size(tuple_type_tail(Is)))
end
function _zip_iterator_size(::Type{Tuple{I}}) where I
    IteratorSize(I)
end
function _zip_iterator_size(::Type{Tuple{}})
    IsInfinite()
end
function IteratorEltype(::Type{Zip{Is}}) where Is <: Tuple
    _zip_iterator_eltype(Is)
end
function _zip_iterator_eltype(::Type{Is}) where Is <: Tuple
    and_iteratoreltype(IteratorEltype(tuple_type_head(Is)), _zip_iterator_eltype(tuple_type_tail(Is)))
end
function _zip_iterator_eltype(::Type{Tuple{}})
    HasEltype()
end

function reverse(z::Zip)
    Zip(map(reverse, z.is))
end

# filter

struct Filter{F,I}
    flt::F
    itr::I
end

"""
    Iterators.filter(flt, itr)

Given a predicate function `flt` and an iterable object `itr`, return an
iterable object which upon iteration yields the elements `x` of `itr` that
satisfy `flt(x)`. The order of the original iterator is preserved.

This function is *lazy*; that is, it is guaranteed to return in ``Θ(1)`` time
and use ``Θ(1)`` additional space, and `flt` will not be called by an
invocation of `filter`. Calls to `flt` will be made when iterating over the
returned iterable object. These calls are not cached and repeated calls will be
made when reiterating.

See [`Base.filter`](@ref) for an eager implementation of filtering for arrays.

# Examples
```jldoctest
julia> f = Iterators.filter(isodd, [1, 2, 3, 4, 5])
Base.Iterators.Filter{typeof(isodd),Array{Int64,1}}(isodd, [1, 2, 3, 4, 5])

julia> foreach(println, f)
1
3
5
```
"""
filter(flt, itr) = Filter(flt, itr)

function iterate(f::Filter, state...)
    y = iterate(f.itr, state...)
    while y !== nothing
        if f.flt(y[1])
            return y
        end
        y = iterate(f.itr, y[2])
    end
    nothing
end

function eltype(::Type{Filter{F, I}}) where {F, I}
    eltype(I)
end
function IteratorEltype(::Type{Filter{F, I}}) where {F, I}
    IteratorEltype(I)
end
function IteratorSize(::Type{<:Filter})
    SizeUnknown()
end

function reverse(f::Filter)
    Filter(f.flt, reverse(f.itr))
end

# Rest -- iterate starting at the given state

struct Rest{I,S}
    itr::I
    st::S
end

"""
    rest(iter, state)

An iterator that yields the same elements as `iter`, but starting at the given `state`.

# Examples
```jldoctest
julia> collect(Iterators.rest([1,2,3,4], 2))
3-element Array{Int64,1}:
 2
 3
 4
```
"""
rest(itr,state) = Rest(itr,state)
function rest(itr::Rest, state)
    Rest(itr.itr, state)
end
function rest(itr)
    itr
end

"""
    peel(iter)

Returns the first element and an iterator over the remaining elements.

# Examples
```jldoctest
julia> (a, rest) = Iterators.peel("abc");

julia> a
'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

julia> collect(rest)
2-element Array{Char,1}:
 'b'
 'c'
```
"""
function peel(itr)
    y = iterate(itr)
    y === nothing && throw(BoundsError())
    val, s = y
    val, rest(itr, s)
end

@propagate_inbounds iterate(i::Rest, st=i.st) = iterate(i.itr, st)
function isdone(i::Rest, st...)
    isdone(i.itr, st...)
end

function eltype(::Type{<:Rest{I}}) where I
    eltype(I)
end
function IteratorEltype(::Type{<:Rest{I}}) where I
    IteratorEltype(I)
end
function rest_iteratorsize(a)
    SizeUnknown()
end
function rest_iteratorsize(::IsInfinite)
    IsInfinite()
end
function IteratorSize(::Type{<:Rest{I}}) where I
    rest_iteratorsize(IteratorSize(I))
end

# Count -- infinite counting

struct Count{S<:Number}
    start::S
    step::S
end

"""
    countfrom(start=1, step=1)

An iterator that counts forever, starting at `start` and incrementing by `step`.

# Examples
```jldoctest
julia> for v in Iterators.countfrom(5, 2)
           v > 10 && break
           println(v)
       end
5
7
9
```
"""
countfrom(start::Number, step::Number) = Count(promote(start, step)...)
function countfrom(start::Number)
    Count(start, oneunit(start))
end
function countfrom()
    Count(1, 1)
end

function eltype(::Type{Count{S}}) where S
    S
end

function iterate(it::Count, state=it.start)
    (state, state + it.step)
end

function IteratorSize(::Type{<:Count})
    IsInfinite()
end

# Take -- iterate through the first n elements

struct Take{I}
    xs::I
    n::Int
    function Take(xs::I, n::Integer) where {I}
        n < 0 && throw(ArgumentError("Take length must be nonnegative"))
        return new{I}(xs, n)
    end
end

"""
    take(iter, n)

An iterator that generates at most the first `n` elements of `iter`.

# Examples
```jldoctest
julia> a = 1:2:11
1:2:11

julia> collect(a)
6-element Array{Int64,1}:
  1
  3
  5
  7
  9
 11

julia> collect(Iterators.take(a,3))
3-element Array{Int64,1}:
 1
 3
 5
```
"""
take(xs, n::Integer) = Take(xs, Int(n))
function take(xs::Take, n::Integer)
    Take(xs.xs, min(Int(n), xs.n))
end

function eltype(::Type{Take{I}}) where I
    eltype(I)
end
function IteratorEltype(::Type{Take{I}}) where I
    IteratorEltype(I)
end
function take_iteratorsize(a)
    HasLength()
end
function take_iteratorsize(::SizeUnknown)
    SizeUnknown()
end
function IteratorSize(::Type{Take{I}}) where I
    take_iteratorsize(IteratorSize(I))
end
function length(t::Take)
    _min_length(t.xs, 1:t.n, IteratorSize(t.xs), HasLength())
end
function isdone(t::Take)
    isdone(t.xs)
end
function isdone(t::Take, state)
    (state[1] <= 0) | isdone(t.xs, tail(state))
end

@propagate_inbounds function iterate(it::Take, state=(it.n,))
    n, rest = state[1], tail(state)
    n <= 0 && return nothing
    y = iterate(it.xs, rest...)
    y === nothing && return nothing
    return y[1], (n - 1, y[2])
end

# Drop -- iterator through all but the first n elements

struct Drop{I}
    xs::I
    n::Int
    function Drop(xs::I, n::Integer) where {I}
        n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
        return new{I}(xs, n)
    end
end

"""
    drop(iter, n)

An iterator that generates all but the first `n` elements of `iter`.

# Examples
```jldoctest
julia> a = 1:2:11
1:2:11

julia> collect(a)
6-element Array{Int64,1}:
  1
  3
  5
  7
  9
 11

julia> collect(Iterators.drop(a,4))
2-element Array{Int64,1}:
  9
 11
```
"""
drop(xs, n::Integer) = Drop(xs, Int(n))
function drop(xs::Take, n::Integer)
    Take(drop(xs.xs, Int(n)), max(0, xs.n - Int(n)))
end
function drop(xs::Drop, n::Integer)
    Drop(xs.xs, Int(n) + xs.n)
end

function eltype(::Type{Drop{I}}) where I
    eltype(I)
end
function IteratorEltype(::Type{Drop{I}}) where I
    IteratorEltype(I)
end
function drop_iteratorsize(::SizeUnknown)
    SizeUnknown()
end
function drop_iteratorsize(::Union{HasShape, HasLength})
    HasLength()
end
function drop_iteratorsize(::IsInfinite)
    IsInfinite()
end
function IteratorSize(::Type{Drop{I}}) where I
    drop_iteratorsize(IteratorSize(I))
end
function length(d::Drop)
    _diff_length(d.xs, 1:d.n, IteratorSize(d.xs), HasLength())
end

function iterate(it::Drop)
    y = iterate(it.xs)
    for i in 1:it.n
        y === nothing && return y
        y = iterate(it.xs, y[2])
    end
    y
end
function iterate(it::Drop, state)
    iterate(it.xs, state)
end
function isdone(it::Drop, state)
    isdone(it.xs, state)
end

# Cycle an iterator forever

struct Cycle{I}
    xs::I
end

"""
    cycle(iter)

An iterator that cycles through `iter` forever.
If `iter` is empty, so is `cycle(iter)`.

# Examples
```jldoctest
julia> for (i, v) in enumerate(Iterators.cycle("hello"))
           print(v)
           i > 10 && break
       end
hellohelloh
```
"""
cycle(xs) = Cycle(xs)

function eltype(::Type{Cycle{I}}) where I
    eltype(I)
end
function IteratorEltype(::Type{Cycle{I}}) where I
    IteratorEltype(I)
end
function IteratorSize(::Type{Cycle{I}}) where I
    IsInfinite()
end

function iterate(it::Cycle)
    iterate(it.xs)
end
function isdone(it::Cycle)
    isdone(it.xs)
end
function isdone(it::Cycle, state)
    false
end
function iterate(it::Cycle, state)
    y = iterate(it.xs, state)
    y === nothing && return iterate(it)
    y
end

function reverse(it::Cycle)
    Cycle(reverse(it.xs))
end

# Repeated - repeat an object infinitely many times

struct Repeated{O}
    x::O
end
function repeated(x)
    Repeated(x)
end

"""
    repeated(x[, n::Int])

An iterator that generates the value `x` forever. If `n` is specified, generates `x` that
many times (equivalent to `take(repeated(x), n)`).

# Examples
```jldoctest
julia> a = Iterators.repeated([1 2], 4);

julia> collect(a)
4-element Array{Array{Int64,2},1}:
 [1 2]
 [1 2]
 [1 2]
 [1 2]
```
"""
repeated(x, n::Integer) = take(repeated(x), Int(n))

function eltype(::Type{Repeated{O}}) where O
    O
end

function iterate(it::Repeated, state...)
    (it.x, nothing)
end

function IteratorSize(::Type{<:Repeated})
    IsInfinite()
end
function IteratorEltype(::Type{<:Repeated})
    HasEltype()
end

function reverse(it::Union{Repeated, Take{<:Repeated}})
    it
end

# Product -- cartesian product of iterators
struct ProductIterator{T<:Tuple}
    iterators::T
end

"""
    product(iters...)

Return an iterator over the product of several iterators. Each generated element is
a tuple whose `i`th element comes from the `i`th argument iterator. The first iterator
changes the fastest.

# Examples
```jldoctest
julia> collect(Iterators.product(1:2, 3:5))
2×3 Array{Tuple{Int64,Int64},2}:
 (1, 3)  (1, 4)  (1, 5)
 (2, 3)  (2, 4)  (2, 5)
```
"""
product(iters...) = ProductIterator(iters)

function IteratorSize(::Type{ProductIterator{Tuple{}}})
    HasShape{0}()
end
function IteratorSize(::Type{ProductIterator{T}}) where T <: Tuple
    prod_iteratorsize(IteratorSize(tuple_type_head(T)), IteratorSize(ProductIterator{tuple_type_tail(T)}))
end

function prod_iteratorsize(::HasLength, ::HasLength)
    HasShape{2}()
end
function prod_iteratorsize(::HasLength, ::HasShape{N}) where N
    HasShape{N + 1}()
end
function prod_iteratorsize(::HasShape{N}, ::HasLength) where N
    HasShape{N + 1}()
end
function prod_iteratorsize(::HasShape{M}, ::HasShape{N}) where {M, N}
    HasShape{M + N}()
end

# products can have an infinite iterator
function prod_iteratorsize(::IsInfinite, ::IsInfinite)
    IsInfinite()
end
function prod_iteratorsize(a, ::IsInfinite)
    IsInfinite()
end
function prod_iteratorsize(::IsInfinite, b)
    IsInfinite()
end
function prod_iteratorsize(a, b)
    SizeUnknown()
end

function size(P::ProductIterator)
    _prod_size(P.iterators)
end
function _prod_size(::Tuple{})
    ()
end
function _prod_size(t::Tuple)
    (_prod_size1(t[1], IteratorSize(t[1]))..., _prod_size(tail(t))...)
end
function _prod_size1(a, ::HasShape)
    size(a)
end
function _prod_size1(a, ::HasLength)
    (length(a),)
end
function _prod_size1(a, A)
    throw(ArgumentError("Cannot compute size for object of type $(typeof(a))"))
end

function axes(P::ProductIterator)
    _prod_indices(P.iterators)
end
function _prod_indices(::Tuple{})
    ()
end
function _prod_indices(t::Tuple)
    (_prod_axes1(t[1], IteratorSize(t[1]))..., _prod_indices(tail(t))...)
end
function _prod_axes1(a, ::HasShape)
    axes(a)
end
function _prod_axes1(a, ::HasLength)
    (OneTo(length(a)),)
end
function _prod_axes1(a, A)
    throw(ArgumentError("Cannot compute indices for object of type $(typeof(a))"))
end

function ndims(p::ProductIterator)
    length(axes(p))
end
function length(P::ProductIterator)
    prod(size(P))
end

function IteratorEltype(::Type{ProductIterator{Tuple{}}})
    HasEltype()
end
function IteratorEltype(::Type{ProductIterator{Tuple{I}}}) where I
    IteratorEltype(I)
end
function IteratorEltype(::Type{ProductIterator{T}}) where {T<:Tuple}
    I = tuple_type_head(T)
    P = ProductIterator{tuple_type_tail(T)}
    IteratorEltype(I) == EltypeUnknown() ? EltypeUnknown() : IteratorEltype(P)
end

function eltype(::Type{<:ProductIterator{I}}) where I
    _prod_eltype(I)
end
function _prod_eltype(::Type{Tuple{}})
    Tuple{}
end
function _prod_eltype(::Type{I}) where I <: Tuple
    Base.tuple_type_cons(eltype(tuple_type_head(I)), _prod_eltype(tuple_type_tail(I)))
end

function iterate(::ProductIterator{Tuple{}})
    ((), true)
end
function iterate(::ProductIterator{Tuple{}}, state)
    nothing
end

@inline isdone(P::ProductIterator) = any(isdone, P.iterators)
@inline function _pisdone(iters, states)
    iter1 = first(iters)
    done1 = isdone(iter1, first(states)[2]) # check step
    done1 === true || return done1 # false or missing
    done1 = isdone(iter1) # check restart
    done1 === true || return done1 # false or missing
    return _pisdone(tail(iters), tail(states)) # check tail
end
@inline isdone(P::ProductIterator, states) = _pisdone(P.iterators, states)

@inline _piterate() = ()
@inline function _piterate(iter1, rest...)
    next = iterate(iter1)
    next === nothing && return nothing
    restnext = _piterate(rest...)
    restnext === nothing && return nothing
    return (next, restnext...)
end
@inline function iterate(P::ProductIterator)
    isdone(P) === true && return nothing
    next = _piterate(P.iterators...)
    next === nothing && return nothing
    return (map(x -> x[1], next), next)
end

@inline _piterate1(::Tuple{}, ::Tuple{}) = nothing
@inline function _piterate1(iters, states)
    iter1 = first(iters)
    next = iterate(iter1, first(states)[2])
    restnext = tail(states)
    if next === nothing
        isdone(iter1) === true && return nothing
        restnext = _piterate1(tail(iters), restnext)
        restnext === nothing && return nothing
        next = iterate(iter1)
        next === nothing && return nothing
    end
    return (next, restnext...)
end
@inline function iterate(P::ProductIterator, states)
    isdone(P, states) === true && return nothing
    next = _piterate1(P.iterators, states)
    next === nothing && return nothing
    return (map(x -> x[1], next), next)
end

function reverse(p::ProductIterator)
    ProductIterator(map(reverse, p.iterators))
end

# flatten an iterator of iterators

struct Flatten{I}
    it::I
end

"""
    flatten(iter)

Given an iterator that yields iterators, return an iterator that yields the
elements of those iterators.
Put differently, the elements of the argument iterator are concatenated.

# Examples
```jldoctest
julia> collect(Iterators.flatten((1:2, 8:9)))
4-element Array{Int64,1}:
 1
 2
 8
 9
```
"""
flatten(itr) = Flatten(itr)

function eltype(::Type{Flatten{I}}) where I
    eltype(eltype(I))
end
function eltype(::Type{Flatten{Tuple{}}})
    eltype(Tuple{})
end
function IteratorEltype(::Type{Flatten{I}}) where I
    _flatteneltype(I, IteratorEltype(I))
end
function IteratorEltype(::Type{Flatten{Tuple{}}})
    IteratorEltype(Tuple{})
end
function _flatteneltype(I, ::HasEltype)
    IteratorEltype(eltype(I))
end
function _flatteneltype(I, et)
    EltypeUnknown()
end

function flatten_iteratorsize(::Union{HasShape, HasLength}, ::Type{<:NTuple{N, Any}}) where N
    HasLength()
end
function flatten_iteratorsize(::Union{HasShape, HasLength}, ::Type{<:Tuple})
    SizeUnknown()
end
function flatten_iteratorsize(::Union{HasShape, HasLength}, ::Type{<:Number})
    HasLength()
end
function flatten_iteratorsize(a, b)
    SizeUnknown()
end

function _flatten_iteratorsize(sz, ::EltypeUnknown, I)
    SizeUnknown()
end
function _flatten_iteratorsize(sz, ::HasEltype, I)
    flatten_iteratorsize(sz, eltype(I))
end
function _flatten_iteratorsize(sz, ::HasEltype, ::Type{Tuple{}})
    HasLength()
end

function IteratorSize(::Type{Flatten{I}}) where I
    _flatten_iteratorsize(IteratorSize(I), IteratorEltype(I), I)
end

function flatten_length(f, T::Type{<:NTuple{N,Any}}) where {N}
    fieldcount(T)*length(f.it)
end
function flatten_length(f, ::Type{<:Number})
    length(f.it)
end
function flatten_length(f, T)
    throw(ArgumentError("Iterates of the argument to Flatten are not known to have constant length"))
end
function length(f::Flatten{I}) where I
    flatten_length(f, eltype(I))
end
function length(f::Flatten{Tuple{}})
    0
end

@propagate_inbounds function iterate(f::Flatten, state=())
    if state !== ()
        y = iterate(tail(state)...)
        y !== nothing && return (y[1], (state[1], state[2], y[2]))
    end
    x = (state === () ? iterate(f.it) : iterate(f.it, state[1]))
    x === nothing && return nothing
    y = iterate(x[1])
    while y === nothing
         x = iterate(f.it, x[2])
         x === nothing && return nothing
         y = iterate(x[1])
    end
    return y[1], (x[2], x[1], y[2])
end

function reverse(f::Flatten)
    Flatten((reverse(itr) for itr = reverse(f.it)))
end

"""
    partition(collection, n)

Iterate over a collection `n` elements at a time.

# Examples
```jldoctest
julia> collect(Iterators.partition([1,2,3,4,5], 2))
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5]
```
"""
partition(c::T, n::Integer) where {T} = PartitionIterator{T}(c, Int(n))


struct PartitionIterator{T}
    c::T
    n::Int
end

function eltype(::Type{PartitionIterator{T}}) where T
    Vector{eltype(T)}
end
function partition_iteratorsize(::HasShape)
    HasLength()
end
function partition_iteratorsize(isz)
    isz
end
function IteratorSize(::Type{PartitionIterator{T}}) where {T}
    partition_iteratorsize(IteratorSize(T))
end

function IteratorEltype(::Type{<:PartitionIterator{T}}) where T
    IteratorEltype(T)
end

function length(itr::PartitionIterator)
    l = length(itr.c)
    return div(l, itr.n) + ((mod(l, itr.n) > 0) ? 1 : 0)
end

function iterate(itr::PartitionIterator{<:Vector}, state=1)
    state > length(itr.c) && return nothing
    r = min(state + itr.n - 1, length(itr.c))
    return view(itr.c, state:r), r + 1
end

struct IterationCutShort; end

function iterate(itr::PartitionIterator, state...)
    v = Vector{eltype(itr.c)}(undef, itr.n)
    # This is necessary to remember whether we cut the
    # last element short. In such cases, we do return that
    # element, but not the next one
    state === (IterationCutShort(),) && return nothing
    i = 0
    y = iterate(itr.c, state...)
    while y !== nothing
        i += 1
        v[i] = y[1]
        if i >= itr.n
            break
        end
        y = iterate(itr.c, y[2])
    end
    i === 0 && return nothing
    return resize!(v, i), y === nothing ? IterationCutShort() : y[2]
end

"""
    Stateful(itr)

There are several different ways to think about this iterator wrapper:

1. It provides a mutable wrapper around an iterator and
   its iteration state.
2. It turns an iterator-like abstraction into a `Channel`-like
   abstraction.
3. It's an iterator that mutates to become its own rest iterator
   whenever an item is produced.

`Stateful` provides the regular iterator interface. Like other mutable iterators
(e.g. [`Channel`](@ref)), if iteration is stopped early (e.g. by a `break` in a `for` loop),
iteration can be resumed from the same spot by continuing to iterate over the
same iterator object (in contrast, an immutable iterator would restart from the
beginning).

# Examples
```jldoctest
julia> a = Iterators.Stateful("abcdef");

julia> isempty(a)
false

julia> popfirst!(a)
'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

julia> collect(Iterators.take(a, 3))
3-element Array{Char,1}:
 'b'
 'c'
 'd'

julia> collect(a)
2-element Array{Char,1}:
 'e'
 'f'
```

```jldoctest
julia> a = Iterators.Stateful([1,1,1,2,3,4]);

julia> for x in a; x == 1 || break; end

julia> Base.peek(a)
3

julia> sum(a) # Sum the remaining elements
7
```
"""
mutable struct Stateful{T, VS}
    itr::T
    # A bit awkward right now, but adapted to the new iteration protocol
    nextvalstate::Union{VS, Nothing}
    taken::Int
    @inline function Stateful{<:Any, Any}(itr::T) where {T}
        new{T, Any}(itr, iterate(itr), 0)
    end
    @inline function Stateful(itr::T) where {T}
        VS = approx_iter_type(T)
        return new{T, VS}(itr, iterate(itr)::VS, 0)
    end
end

function reset!(s::Stateful{T,VS}, itr::T) where {T,VS}
    s.itr = itr
    setfield!(s, :nextvalstate, iterate(itr))
    s.taken = 0
    s
end

if Base === Core.Compiler
    approx_iter_type(a::Type) = Any
else
    # Try to find an appropriate type for the (value, state tuple),
    # by doing a recursive unrolling of the iteration protocol up to
    # fixpoint.
    approx_iter_type(itrT::Type) = _approx_iter_type(itrT, Base._return_type(iterate, Tuple{itrT}))
    # Not actually called, just passed to return type to avoid
    # having to typesubtract
    function doiterate(itr, valstate::Union{Nothing, Tuple{Any, Any}})
        valstate === nothing && return nothing
        val, st = valstate
        return iterate(itr, st)
    end
    function _approx_iter_type(itrT::Type, vstate::Type)
        vstate <: Union{Nothing, Tuple{Any, Any}} || return Any
        vstate <: Union{} && return Union{}
        nextvstate = Base._return_type(doiterate, Tuple{itrT, vstate})
        return (nextvstate <: vstate ? vstate : Any)
    end
end

function convert(::Type{Stateful}, itr)
    Stateful(itr)
end

@inline isdone(s::Stateful, st=nothing) = s.nextvalstate === nothing

@inline function popfirst!(s::Stateful)
    vs = s.nextvalstate
    if vs === nothing
        throw(EOFError())
    else
        val, state = vs
        Core.setfield!(s, :nextvalstate, iterate(s.itr, state))
        s.taken += 1
        return val
    end
end

@inline peek(s::Stateful, sentinel=nothing) = s.nextvalstate !== nothing ? s.nextvalstate[1] : sentinel
@inline iterate(s::Stateful, state=nothing) = s.nextvalstate === nothing ? nothing : (popfirst!(s), nothing)
function IteratorSize(::Type{Stateful{T, VS}}) where {T, VS}
    if IteratorSize(T) isa HasShape
        HasLength()
    else
        IteratorSize(T)
    end
end
function eltype(::Type{Stateful{T, VS}} where VS) where T
    eltype(T)
end
function IteratorEltype(::Type{Stateful{T, VS}}) where {T, VS}
    IteratorEltype(T)
end
function length(s::Stateful)
    length(s.itr) - s.taken
end

end
