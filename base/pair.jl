# This file is a part of Julia. License is MIT: https://julialang.org/license

struct Pair{A, B}
    first::A
    second::B
    function Pair{A, B}(@nospecialize(a), @nospecialize(b)) where {A, B}
        @_inline_meta
        # if we didn't inline this, it's probably because the callsite was actually dynamic
        # to avoid potentially compiling many copies of this, we mark the arguments with `@nospecialize`
        # but also mark the whole function with `@inline` to ensure we will inline it whenever possible
        # (even if `convert(::Type{A}, a::A)` for some reason was expensive)
        return new(a, b)
    end
end
function Pair(a, b)
    Pair{typeof(a), typeof(b)}(a, b)
end
const => = Pair

"""
    Pair(x, y)
    x => y

Construct a `Pair` object with type `Pair{typeof(x), typeof(y)}`. The elements
are stored in the fields `first` and `second`. They can also be accessed via
iteration.

See also: [`Dict`](@ref)

# Examples
```jldoctest
julia> p = "foo" => 7
"foo" => 7

julia> typeof(p)
Pair{String,Int64}

julia> p.first
"foo"

julia> for x in p
           println(x)
       end
foo
7
```
"""
Pair, =>

function eltype(p::Type{Pair{A, B}}) where {A, B}
    Union{A, B}
end
function iterate(p::Pair, i=1)
    if i > 2
        nothing
    else
        (getfield(p, i), i + 1)
    end
end
function indexed_iterate(p::Pair, i::Int, state=1)
    (getfield(p, i), i + 1)
end

function hash(p::Pair, h::UInt)
    hash(p.second, hash(p.first, h))
end

function ==(p::Pair, q::Pair)
    (p.first == q.first) & (p.second == q.second)
end
function isequal(p::Pair, q::Pair)
    isequal(p.first, q.first) & isequal(p.second, q.second)
end

function isless(p::Pair, q::Pair)
    ifelse(!(isequal(p.first, q.first)), isless(p.first, q.first), isless(p.second, q.second))
end
function getindex(p::Pair, i::Int)
    getfield(p, i)
end
function getindex(p::Pair, i::Real)
    getfield(p, convert(Int, i))
end
function reverse(p::Pair{A, B}) where {A, B}
    Pair{B, A}(p.second, p.first)
end

function firstindex(p::Pair)
    1
end
function lastindex(p::Pair)
    2
end
function length(p::Pair)
    2
end
function first(p::Pair)
    p.first
end
function last(p::Pair)
    p.second
end

function convert(::Type{Pair{A, B}}, x::Pair{A, B}) where {A, B}
    x
end
function convert(::Type{Pair{A,B}}, x::Pair) where {A,B}
    Pair{A,B}(convert(A, x[1]), convert(B, x[2]))
end

function promote_rule(::Type{Pair{A1, B1}}, ::Type{Pair{A2, B2}}) where {A1, B1, A2, B2}
    Pair{promote_type(A1, A2), promote_type(B1, B2)}
end
