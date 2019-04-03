# This file is a part of Julia. License is MIT: https://julialang.org/license

module Order


import ..@__MODULE__, ..parentmodule
const Base = parentmodule(@__MODULE__)
import .Base:
    AbstractVector, @propagate_inbounds, isless, identity, getindex,
    +, -, !, &, <, |

## notions of element ordering ##

export # not exported by Base
    Ordering, Forward, Reverse,
    By, Lt, Perm,
    ReverseOrdering, ForwardOrdering,
    DirectOrdering,
    lt, ord, ordtype

abstract type Ordering end

struct ForwardOrdering <: Ordering end
struct ReverseOrdering{Fwd<:Ordering} <: Ordering
    fwd::Fwd
end

function ReverseOrdering(rev::ReverseOrdering)
    rev.fwd
end
function ReverseOrdering(fwd::Fwd) where Fwd
    ReverseOrdering{Fwd}(fwd)
end

const DirectOrdering = Union{ForwardOrdering,ReverseOrdering{ForwardOrdering}}

const Forward = ForwardOrdering()
const Reverse = ReverseOrdering(Forward)

struct By{T} <: Ordering
    by::T
end

struct Lt{T} <: Ordering
    lt::T
end

struct Perm{O<:Ordering,V<:AbstractVector} <: Ordering
    order::O
    data::V
end

function lt(o::ForwardOrdering, a, b)
    isless(a, b)
end
function lt(o::ReverseOrdering, a, b)
    lt(o.fwd, b, a)
end
function lt(o::By, a, b)
    isless(o.by(a), o.by(b))
end
function lt(o::Lt, a, b)
    o.lt(a, b)
end

@propagate_inbounds function lt(p::Perm, a::Integer, b::Integer)
    da = p.data[a]
    db = p.data[b]
    lt(p.order, da, db) | (!lt(p.order, db, da) & (a < b))
end

function ordtype(o::ReverseOrdering, vs::AbstractArray)
    ordtype(o.fwd, vs)
end
function ordtype(o::Perm, vs::AbstractArray)
    ordtype(o.order, o.data)
end
# TODO: here, we really want the return type of o.by, without calling it
function ordtype(o::By, vs::AbstractArray)
    try
        typeof(o.by(vs[1]))
    catch
        Any
    end
end
function ordtype(o::Ordering, vs::AbstractArray)
    eltype(vs)
end

function _ord(lt::typeof(isless), by::typeof(identity), order::Ordering)
    order
end
function _ord(lt::typeof(isless), by, order::Ordering)
    By(by)
end
function _ord(lt, by::typeof(identity), order::Ordering)
    Lt(lt)
end
function _ord(lt, by, order::Ordering)
    Lt(((x, y)->begin
                lt(by(x), by(y))
            end))
end

function ord(lt, by, rev::Nothing, order::Ordering=Forward)
    _ord(lt, by, order)
end

function ord(lt, by, rev::Bool, order::Ordering=Forward)
    o = _ord(lt, by, order)
    return rev ? ReverseOrdering(o) : o
end

end
