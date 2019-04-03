# This file is a part of Julia. License is MIT: https://julialang.org/license

# weak key dictionaries

"""
    WeakKeyDict([itr])

`WeakKeyDict()` constructs a hash table where the keys are weak
references to objects, and thus may be garbage collected even when
referenced in a hash table.

See [`Dict`](@ref) for further help.  Note, unlike [`Dict`](@ref),
`WeakKeyDict` does not convert keys on insertion.
"""
mutable struct WeakKeyDict{K,V} <: AbstractDict{K,V}
    ht::Dict{WeakRef,V}
    lock::ReentrantLock
    finalizer::Function

    # Constructors mirror Dict's
    function WeakKeyDict{K,V}() where V where K
        t = new(Dict{Any,V}(), ReentrantLock(), identity)
        t.finalizer = function (k)
            # when a weak key is finalized, remove from dictionary if it is still there
            if islocked(t)
                finalizer(t.finalizer, k)
                return nothing
            end
            delete!(t, k)
        end
        return t
    end
end
function WeakKeyDict{K,V}(kv) where V where K
    h = WeakKeyDict{K,V}()
    for (k,v) in kv
        h[k] = v
    end
    return h
end
function (WeakKeyDict{K, V}(p::Pair) where V) where K
    setindex!(WeakKeyDict{K, V}(), p.second, p.first)
end
function WeakKeyDict{K,V}(ps::Pair...) where V where K
    h = WeakKeyDict{K,V}()
    sizehint!(h, length(ps))
    for p in ps
        h[p.first] = p.second
    end
    return h
end
function WeakKeyDict()
    WeakKeyDict{Any, Any}()
end

function WeakKeyDict(kv::Tuple{})
    WeakKeyDict()
end
function copy(d::WeakKeyDict)
    WeakKeyDict(d)
end

function WeakKeyDict(ps::Pair{K, V}...) where {K, V}
    WeakKeyDict{K, V}(ps)
end
function WeakKeyDict(ps::Pair{K}...) where K
    WeakKeyDict{K, Any}(ps)
end
function WeakKeyDict(ps::(Pair{K, V} where K)...) where V
    WeakKeyDict{Any, V}(ps)
end
function WeakKeyDict(ps::Pair...)
    WeakKeyDict{Any, Any}(ps)
end

function WeakKeyDict(kv)
    try
        Base.dict_with_eltype((K, V) -> WeakKeyDict{K, V}, kv, eltype(kv))
    catch
        if !isiterable(typeof(kv)) || !all(x->isa(x,Union{Tuple,Pair}),kv)
            throw(ArgumentError("WeakKeyDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow()
        end
    end
end

function empty(d::WeakKeyDict, ::Type{K}, ::Type{V}) where {K, V}
    WeakKeyDict{K, V}()
end

function islocked(wkh::WeakKeyDict)
    islocked(wkh.lock)
end
function lock(f, wkh::WeakKeyDict)
    lock(f, wkh.lock)
end
function trylock(f, wkh::WeakKeyDict)
    trylock(f, wkh.lock)
end

function setindex!(wkh::WeakKeyDict{K}, v, key) where K
    !isa(key, K) && throw(ArgumentError("$(limitrepr(key)) is not a valid key for type $K"))
    finalizer(wkh.finalizer, key)
    lock(wkh) do
        wkh.ht[WeakRef(key)] = v
    end
    return wkh
end

function getkey(wkh::WeakKeyDict{K}, kk, default) where K
    return lock(wkh) do
        k = getkey(wkh.ht, kk, secret_table_token)
        k === secret_table_token && return default
        return k.value::K
    end
end

function map!(f, iter::ValueIterator{<:WeakKeyDict})
    map!(f, values((iter.dict).ht))
end
function get(wkh::WeakKeyDict{K}, key, default) where K
    lock((()->begin
                get(wkh.ht, key, default)
            end), wkh)
end
function get(default::Callable, wkh::WeakKeyDict{K}, key) where K
    lock((()->begin
                get(default, wkh.ht, key)
            end), wkh)
end
function get!(wkh::WeakKeyDict{K}, key, default) where {K}
    !isa(key, K) && throw(ArgumentError("$(limitrepr(key)) is not a valid key for type $K"))
    lock(() -> get!(wkh.ht, WeakRef(key), default), wkh)
end
function get!(default::Callable, wkh::WeakKeyDict{K}, key) where {K}
    !isa(key, K) && throw(ArgumentError("$(limitrepr(key)) is not a valid key for type $K"))
    lock(() -> get!(default, wkh.ht, WeakRef(key)), wkh)
end
function pop!(wkh::WeakKeyDict{K}, key) where K
    lock((()->begin
                pop!(wkh.ht, key)
            end), wkh)
end
function pop!(wkh::WeakKeyDict{K}, key, default) where K
    lock((()->begin
                pop!(wkh.ht, key, default)
            end), wkh)
end
function delete!(wkh::WeakKeyDict, key)
    lock((()->begin
                delete!(wkh.ht, key)
            end), wkh)
end
function empty!(wkh::WeakKeyDict)
    lock((()->begin
                empty!(wkh.ht)
            end), wkh)
    wkh
end
function haskey(wkh::WeakKeyDict{K}, key) where K
    lock((()->begin
                haskey(wkh.ht, key)
            end), wkh)
end
function getindex(wkh::WeakKeyDict{K}, key) where K
    lock((()->begin
                getindex(wkh.ht, key)
            end), wkh)
end
function isempty(wkh::WeakKeyDict)
    isempty(wkh.ht)
end
function length(t::WeakKeyDict)
    length(t.ht)
end

function iterate(t::WeakKeyDict{K,V}) where V where K
    gc_token = Ref{Bool}(false) # no keys will be deleted via finalizers until this token is gc'd
    finalizer(gc_token) do r
        if r[]
            r[] = false
            unlock(t.lock)
        end
    end
    s = lock(t.lock)
    iterate(t, (gc_token,))
end
function iterate(t::WeakKeyDict{K,V}, state) where V where K
    gc_token = first(state)
    y = iterate(t.ht, tail(state)...)
    y === nothing && return nothing
    wkv, i = y
    kv = Pair{K,V}(wkv[1].value::K, wkv[2])
    return (kv, (gc_token, i))
end

function filter!(f, d::WeakKeyDict)
    filter_in_one_pass!(f, d)
end
