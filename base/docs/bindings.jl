# This file is a part of Julia. License is MIT: https://julialang.org/license

export @var

struct Binding
    mod::Module
    var::Symbol

    function Binding(m::Module, v::Symbol)
        # Normalise the binding module for module symbols so that:
        #   Binding(Base, :Base) === Binding(Main, :Base)
        m = nameof(m) === v ? parentmodule(m) : m
        new(Base.binding_module(m, v), v)
    end
end

function bindingexpr(x)
    Expr(:call, Binding, splitexpr(x)...)
end

function defined(b::Binding)
    isdefined(b.mod, b.var)
end
function resolve(b::Binding)
    getfield(b.mod, b.var)
end

function splitexpr(x::Expr)
    isexpr(x, :macrocall) ? splitexpr(x.args[1]) :
    isexpr(x, :.)         ? (x.args[1], x.args[2]) :
    error("Invalid @var syntax `$x`.")
end
function splitexpr(s::Symbol)
    (Expr(:macrocall, getfield(Base, Symbol("@__MODULE__")), nothing), quot(s))
end
function splitexpr(other)
    error("Invalid @var syntax `$(other)`.")
end

macro var(x)
    esc(bindingexpr(x))
end

function Base.show(io::IO, b::Binding)
    if b.mod === Main
        print(io, b.var)
    else
        print(io, b.mod, '.', Base.isoperator(b.var) ? ":" : "", b.var)
    end
end

function aliasof(b::Binding)
    if defined(b)
        a = aliasof(resolve(b), b)
        if defined(a)
            a
        else
            b
        end
    else
        b
    end
end
function aliasof(d::DataType, b)
    Binding((d.name).module, (d.name).name)
end
function aliasof(λ::Function, b)
    m = ((typeof(λ)).name).mt
    Binding(m.module, m.name)
end
function aliasof(m::Module, b)
    Binding(m, nameof(m))
end
function aliasof(other, b)
    b
end
