# This file is a part of Julia. License is MIT: https://julialang.org/license

# commented-out definitions are implemented in C

#abstract type Any <: Any end
#abstract type Type{T} end

#abstract type Vararg{T} end

#mutable struct Symbol
#    #opaque
#end

#mutable struct TypeName
#    name::Symbol
#end

#mutable struct DataType <: Type
#    name::TypeName
#    super::Type
#    parameters::Tuple
#    names::Tuple
#    types::Tuple
#    ctor
#    instance
#    size::Int32
#    abstract::Bool
#    mutable::Bool
#    pointerfree::Bool
#end

#struct Union <: Type
#    a
#    b
#end

#mutable struct TypeVar
#    name::Symbol
#    lb::Type
#    ub::Type
#end

#struct UnionAll
#    var::TypeVar
#    body
#end

#struct Nothing
#end
#const nothing = Nothing()

#abstract type AbstractArray{T,N} end
#abstract type DenseArray{T,N} <: AbstractArray{T,N} end

#mutable struct Array{T,N} <: DenseArray{T,N}
#end

#mutable struct Module
#    name::Symbol
#end

#mutable struct Method
#end

#mutable struct MethodInstance
#end

#mutable struct CodeInfo
#end

#mutable struct TypeMapLevel
#end

#mutable struct TypeMapEntry
#end

#abstract type Ref{T} end
#primitive type Ptr{T} <: Ref{T} {32|64} end

# types for the front end

#mutable struct Expr
#    head::Symbol
#    args::Array{Any,1}
#end

#struct LineNumberNode
#    line::Int
#    file::Any # nominally Union{Symbol,Nothing}
#end

#struct LineInfoNode
#    method::Any
#    file::Symbol
#    line::Int
#    inlined_at::Int
#end

#struct GotoNode
#    label::Int
#end

#struct PiNode
#    val
#    typ
#end

#struct PhiNode
#    edges::Vector{Any}
#    values::Vector{Any}
#end

#struct PhiCNode
#    values::Vector{Any}
#end

#struct UpsilonNode
#    val
#end

#struct QuoteNode
#    value
#end

#struct GlobalRef
#    mod::Module
#    name::Symbol
#end

#mutable struct Task
#    parent::Task
#    storage::Any
#    state::Symbol
#    donenotify::Any
#    result::Any
#    exception::Any
#    backtrace::Any
#    logstate::Any
#    code::Any
#end

export
    # key types
    Any, DataType, Vararg, NTuple,
    Tuple, Type, UnionAll, TypeVar, Union, Nothing, Cvoid,
    AbstractArray, DenseArray, NamedTuple,
    # special objects
    Function, Method,
    Module, Symbol, Task, Array, UndefInitializer, undef, WeakRef, VecElement,
    # numeric types
    Number, Real, Integer, Bool, Ref, Ptr,
    AbstractFloat, Float16, Float32, Float64,
    Signed, Int, Int8, Int16, Int32, Int64, Int128,
    Unsigned, UInt, UInt8, UInt16, UInt32, UInt64, UInt128,
    # string types
    AbstractChar, Char, AbstractString, String, IO,
    # errors
    ErrorException, BoundsError, DivideError, DomainError, Exception,
    InterruptException, InexactError, OutOfMemoryError, ReadOnlyMemoryError,
    OverflowError, StackOverflowError, SegmentationFault, UndefRefError, UndefVarError,
    TypeError, ArgumentError, MethodError, AssertionError, LoadError, InitError,
    UndefKeywordError,
    # AST representation
    Expr, QuoteNode, LineNumberNode, GlobalRef,
    # object model functions
    fieldtype, getfield, setfield!, nfields, throw, tuple, ===, isdefined, eval, ifelse,
    # sizeof    # not exported, to avoid conflicting with Base.sizeof
    # type reflection
    <:, typeof, isa, typeassert,
    # method reflection
    applicable, invoke,
    # constants
    nothing, Main

const getproperty = getfield
const setproperty! = setfield!

abstract type Number end
abstract type Real     <: Number end
abstract type AbstractFloat <: Real end
abstract type Integer  <: Real end
abstract type Signed   <: Integer end
abstract type Unsigned <: Integer end

primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end

#primitive type Bool <: Integer 8 end
abstract type AbstractChar end
primitive type Char <: AbstractChar 32 end

primitive type Int8    <: Signed   8 end
#primitive type UInt8   <: Unsigned 8 end
primitive type Int16   <: Signed   16 end
primitive type UInt16  <: Unsigned 16 end
#primitive type Int32   <: Signed   32 end
#primitive type UInt32  <: Unsigned 32 end
#primitive type Int64   <: Signed   64 end
#primitive type UInt64  <: Unsigned 64 end
primitive type Int128  <: Signed   128 end
primitive type UInt128 <: Unsigned 128 end

if Int === Int64
    const UInt = UInt64
else
    const UInt = UInt32
end

function Typeof end
ccall(:jl_toplevel_eval_in, Any, (Any, Any),
      Core, quote
      (f::typeof(Typeof))(x) = ($(_expr(:meta,:nospecialize,:x)); isa(x,Type) ? Type{x} : typeof(x))
      end)

macro nospecialize(x)
    _expr(:meta, :nospecialize, x)
end

function Expr(@nospecialize(args...))
    _expr(args...)
end

abstract type Exception end
struct ErrorException <: Exception
    msg::AbstractString
end

macro _inline_meta()
    Expr(:meta, :inline)
end

macro _noinline_meta()
    Expr(:meta, :noinline)
end

struct BoundsError <: Exception
    a::Any
    i::Any
    BoundsError() = new()
    BoundsError(@nospecialize(a)) = (@_noinline_meta; new(a))
    BoundsError(@nospecialize(a), i) = (@_noinline_meta; new(a,i))
end
struct DivideError         <: Exception end
struct OutOfMemoryError    <: Exception end
struct ReadOnlyMemoryError <: Exception end
struct SegmentationFault   <: Exception end
struct StackOverflowError  <: Exception end
struct UndefRefError       <: Exception end
struct UndefVarError <: Exception
    var::Symbol
end
struct InterruptException <: Exception end
struct DomainError <: Exception
    val
    msg::AbstractString
    DomainError(@nospecialize(val)) = (@_noinline_meta; new(val, ""))
    DomainError(@nospecialize(val), @nospecialize(msg)) = (@_noinline_meta; new(val, msg))
end
struct TypeError <: Exception
    # `func` is the name of the builtin function that encountered a type error,
    # the name of the type that hit an error in its definition or application, or
    # some other brief description of where the error happened.
    # `context` optionally adds extra detail, e.g. the name of the type parameter
    # that got a bad value.
    func::Symbol
    context::Union{AbstractString,Symbol}
    expected::Type
    got
    TypeError(func, context, @nospecialize(expected::Type), @nospecialize(got)) =
        new(func, context, expected, got)
end
function TypeError(where, @nospecialize(expected::Type), @nospecialize(got))
    TypeError(Symbol(where), "", expected, got)
end
struct InexactError <: Exception
    func::Symbol
    T  # Type
    val
    InexactError(f::Symbol, @nospecialize(T), @nospecialize(val)) = (@_noinline_meta; new(f, T, val))
end
struct OverflowError <: Exception
    msg::AbstractString
end

struct ArgumentError <: Exception
    msg::AbstractString
end
struct UndefKeywordError <: Exception
    var::Symbol
end

struct MethodError <: Exception
    f
    args
    world::UInt
    MethodError(@nospecialize(f), @nospecialize(args), world::UInt) = new(f, args, world)
end
const typemax_UInt = ccall(:jl_typemax_uint, Any, (Any,), UInt)
function MethodError(@nospecialize(f), @nospecialize(args))
    MethodError(f, args, typemax_UInt)
end

struct AssertionError <: Exception
    msg::AbstractString
end
function AssertionError()
    AssertionError("")
end

abstract type WrappedException <: Exception end

struct LoadError <: WrappedException
    file::AbstractString
    line::Int
    error
end

struct InitError <: WrappedException
    mod::Symbol
    error
end

function String(s::String)
    s
end  # no constructor yet

const Cvoid = Nothing
function Nothing()
    nothing
end

# This should always be inlined
function getptls()
    ccall(:jl_get_ptls_states, Ptr{Cvoid}, ())
end

function include(m::Module, fname::String)
    ccall(:jl_load_, Any, (Any, Any), m, fname)
end

function eval(m::Module, @nospecialize(e))
    ccall(:jl_toplevel_eval_in, Any, (Any, Any), m, e)
end

function kwfunc(@nospecialize(f))
    ccall(:jl_get_keyword_sorter, Any, (Any,), f)
end

function kwftype(@nospecialize(t))
    typeof(ccall(:jl_get_kwsorter, Any, (Any,), t))
end

mutable struct Box
    contents::Any
    Box(@nospecialize(x)) = new(x)
    Box() = new()
end

# constructors for built-in types

mutable struct WeakRef
    value
    WeakRef() = WeakRef(nothing)
    WeakRef(@nospecialize(v)) = ccall(:jl_gc_new_weakref_th, Ref{WeakRef},
                                      (Ptr{Cvoid}, Any), getptls(), v)
end

function TypeVar(n::Symbol)
    _typevar(n, Union{}, Any)
end
function TypeVar(n::Symbol, @nospecialize(ub))
    _typevar(n, Union{}, ub)
end
function TypeVar(n::Symbol, @nospecialize(lb), @nospecialize(ub))
    _typevar(n, lb, ub)
end

function UnionAll(v::TypeVar, @nospecialize(t))
    ccall(:jl_type_unionall, Any, (Any, Any), v, t)
end

function (::Type{Tuple{}})()
    ()
end # Tuple{}()

struct VecElement{T}
    value::T
    VecElement{T}(value::T) where {T} = new(value) # disable converting constructor in Core
end
function VecElement(arg::T) where T
    VecElement{T}(arg)
end

function _new(typ::Symbol, argty::Symbol)
    eval(Core, $(Expr(:quote, :(($(Expr(:$, :typ)))(@nospecialize(n::$(Expr(:$, :argty)))) = begin
          $(Expr(:$, :(Expr(:new, typ, :n))))
      end))))
end
_new(:GotoNode, :Int)
_new(:NewvarNode, :SlotNumber)
_new(:QuoteNode, :Any)
_new(:SSAValue, :Int)
eval(Core, :(LineNumberNode(l::Int) = $(Expr(:new, :LineNumberNode, :l, nothing))))
eval(Core, :(LineNumberNode(l::Int, @nospecialize(f)) = $(Expr(:new, :LineNumberNode, :l, :f))))
eval(Core, :(GlobalRef(m::Module, s::Symbol) = $(Expr(:new, :GlobalRef, :m, :s))))
eval(Core, :(SlotNumber(n::Int) = $(Expr(:new, :SlotNumber, :n))))
eval(Core, :(TypedSlot(n::Int, @nospecialize(t)) = $(Expr(:new, :TypedSlot, :n, :t))))
eval(Core, :(PhiNode(edges::Array{Any, 1}, values::Array{Any, 1}) = $(Expr(:new, :PhiNode, :edges, :values))))
eval(Core, :(PiNode(val, typ) = $(Expr(:new, :PiNode, :val, :typ))))
eval(Core, :(PhiCNode(values::Array{Any, 1}) = $(Expr(:new, :PhiCNode, :values))))
eval(Core, :(UpsilonNode(val) = $(Expr(:new, :UpsilonNode, :val))))
eval(Core, :(UpsilonNode() = $(Expr(:new, :UpsilonNode))))
eval(Core, :(LineInfoNode(@nospecialize(method), file::Symbol, line::Int, inlined_at::Int) =
             $(Expr(:new, :LineInfoNode, :method, :file, :line, :inlined_at))))

function Module(name::Symbol=:anonymous, std_imports::Bool=true)
    ccall(:jl_f_new_module, Ref{Module}, (Any, Bool), name, std_imports)
end

function _Task(@nospecialize(f), reserved_stack::Int, completion_future)
    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), f, completion_future, reserved_stack)
end

# simple convert for use by constructors of types in Core
# note that there is no actual conversion defined here,
# so the methods and ccall's in Core aren't permitted to use convert
function convert(::Type{Any}, @nospecialize(x))
    x
end
function convert(::Type{T}, x::T) where T
    x
end
function cconvert(::Type{T}, x) where T
    convert(T, x)
end
function unsafe_convert(::Type{T}, x::T) where T
    x
end

const NTuple{N,T} = Tuple{Vararg{T,N}}


## primitive Array constructors
struct UndefInitializer end
const undef = UndefInitializer()
# type and dimensionality specified, accepting dims as series of Ints
function Array{T, 1}(::UndefInitializer, m::Int) where T
    ccall(:jl_alloc_array_1d, Array{T, 1}, (Any, Int), Array{T, 1}, m)
end
function Array{T, 2}(::UndefInitializer, m::Int, n::Int) where T
    ccall(:jl_alloc_array_2d, Array{T, 2}, (Any, Int, Int), Array{T, 2}, m, n)
end
function Array{T, 3}(::UndefInitializer, m::Int, n::Int, o::Int) where T
    ccall(:jl_alloc_array_3d, Array{T, 3}, (Any, Int, Int, Int), Array{T, 3}, m, n, o)
end
function Array{T, N}(::UndefInitializer, d::Vararg{Int, N}) where {T, N}
    ccall(:jl_new_array, Array{T, N}, (Any, Any), Array{T, N}, d)
end
# type and dimensionality specified, accepting dims as tuples of Ints
function Array{T, 1}(::UndefInitializer, d::NTuple{1, Int}) where T
    Array{T, 1}(undef, getfield(d, 1))
end
function Array{T, 2}(::UndefInitializer, d::NTuple{2, Int}) where T
    Array{T, 2}(undef, getfield(d, 1), getfield(d, 2))
end
function Array{T, 3}(::UndefInitializer, d::NTuple{3, Int}) where T
    Array{T, 3}(undef, getfield(d, 1), getfield(d, 2), getfield(d, 3))
end
function Array{T, N}(::UndefInitializer, d::NTuple{N, Int}) where {T, N}
    ccall(:jl_new_array, Array{T, N}, (Any, Any), Array{T, N}, d)
end
# type but not dimensionality specified
function Array{T}(::UndefInitializer, m::Int) where T
    Array{T, 1}(undef, m)
end
function Array{T}(::UndefInitializer, m::Int, n::Int) where T
    Array{T, 2}(undef, m, n)
end
function Array{T}(::UndefInitializer, m::Int, n::Int, o::Int) where T
    Array{T, 3}(undef, m, n, o)
end
function Array{T}(::UndefInitializer, d::NTuple{N, Int}) where {T, N}
    Array{T, N}(undef, d)
end
# empty vector constructor
function Array{T, 1}() where T
    Array{T, 1}(undef, 0)
end


function (::Type{Array{T, N} where T})(x::AbstractArray{S, N}) where {S, N}
    Array{S, N}(x)
end

function Array(A::AbstractArray{T, N}) where {T, N}
    Array{T, N}(A)
end
function Array{T}(A::AbstractArray{S, N}) where {T, N, S}
    Array{T, N}(A)
end

function AbstractArray{T}(A::AbstractArray{S, N}) where {T, S, N}
    AbstractArray{T, N}(A)
end

# primitive Symbol constructors
function Symbol(s::String)
    return ccall(:jl_symbol_n, Ref{Symbol}, (Ptr{UInt8}, Int),
                 ccall(:jl_string_ptr, Ptr{UInt8}, (Any,), s),
                 sizeof(s))
end
function Symbol(a::Array{UInt8,1})
    return ccall(:jl_symbol_n, Ref{Symbol}, (Ptr{UInt8}, Int),
                 ccall(:jl_array_ptr, Ptr{UInt8}, (Any,), a),
                 Intrinsics.arraylen(a))
end
function Symbol(s::Symbol)
    s
end

# module providing the IR object model
module IR
export CodeInfo, MethodInstance, GotoNode,
    NewvarNode, SSAValue, Slot, SlotNumber, TypedSlot,
    PiNode, PhiNode, PhiCNode, UpsilonNode, LineInfoNode

import Core: CodeInfo, MethodInstance, GotoNode,
    NewvarNode, SSAValue, Slot, SlotNumber, TypedSlot,
    PiNode, PhiNode, PhiCNode, UpsilonNode, LineInfoNode

end

# docsystem basics
const unescape = Symbol("hygienic-scope")
macro doc(x...)
    docex = atdoc(__source__, __module__, x...)
    isa(docex, Expr) && docex.head === :escape && return docex
    return Expr(:escape, Expr(unescape, docex, typeof(atdoc).name.module))
end
macro __doc__(x)
    return Expr(:escape, Expr(:block, Expr(:meta, :doc), x))
end
atdoc     = (source, mod, str, expr) -> Expr(:escape, expr)
function atdoc!(λ)
    global atdoc = λ
end

# macros for big integer syntax
macro int128_str end
macro uint128_str end
macro big_str end

# macro for command syntax
macro cmd end


# simple stand-alone print definitions for debugging
abstract type IO end
struct CoreSTDOUT <: IO end
struct CoreSTDERR <: IO end
const stdout = CoreSTDOUT()
const stderr = CoreSTDERR()
function io_pointer(::CoreSTDOUT)
    Intrinsics.pointerref(Intrinsics.cglobal(:jl_uv_stdout, Ptr{Cvoid}), 1, 1)
end
function io_pointer(::CoreSTDERR)
    Intrinsics.pointerref(Intrinsics.cglobal(:jl_uv_stderr, Ptr{Cvoid}), 1, 1)
end

function unsafe_write(io::IO, x::Ptr{UInt8}, nb::UInt)
    ccall(:jl_uv_puts, Cvoid, (Ptr{Cvoid}, Ptr{UInt8}, UInt), io_pointer(io), x, nb)
    nb
end
function unsafe_write(io::IO, x::Ptr{UInt8}, nb::Int)
    ccall(:jl_uv_puts, Cvoid, (Ptr{Cvoid}, Ptr{UInt8}, Int), io_pointer(io), x, nb)
    nb
end
function write(io::IO, x::UInt8)
    ccall(:jl_uv_putb, Cvoid, (Ptr{Cvoid}, UInt8), io_pointer(io), x)
    1
end
function write(io::IO, x::String)
    nb = sizeof(x)
    unsafe_write(io, ccall(:jl_string_ptr, Ptr{UInt8}, (Any,), x), nb)
    return nb
end

function show(io::IO, @nospecialize(x))
    ccall(:jl_static_show, Cvoid, (Ptr{Cvoid}, Any), io_pointer(io), x)
end
function print(io::IO, x::AbstractChar)
    ccall(:jl_uv_putc, Cvoid, (Ptr{Cvoid}, Char), io_pointer(io), x)
end
function print(io::IO, x::String)
    write(io, x)
    nothing
end
function print(io::IO, @nospecialize(x))
    show(io, x)
end
function print(io::IO, @nospecialize(x), @nospecialize(a...))
    print(io, x)
    print(io, a...)
end
function println(io::IO)
    write(io, 0x0a)
    nothing
end # 0x0a = '\n'
function println(io::IO, @nospecialize(x...))
    print(io, x...)
    println(io)
end

function show(@nospecialize(a))
    show(stdout, a)
end
function print(@nospecialize(a...))
    print(stdout, a...)
end
function println(@nospecialize(a...))
    println(stdout, a...)
end

struct GeneratedFunctionStub
    gen
    argnames::Array{Any,1}
    spnames::Union{Nothing, Array{Any,1}}
    line::Int
    file::Symbol
    expand_early::Bool
end

# invoke and wrap the results of @generated
function (g::GeneratedFunctionStub)(@nospecialize args...)
    body = g.gen(args...)
    if body isa CodeInfo
        return body
    end
    lam = Expr(:lambda, g.argnames,
               Expr(Symbol("scope-block"),
                    Expr(:block,
                         LineNumberNode(g.line, g.file),
                         Expr(:meta, :push_loc, g.file, Symbol("@generated body")),
                         Expr(:return, body),
                         Expr(:meta, :pop_loc))))
    if g.spnames === nothing
        return lam
    else
        return Expr(Symbol("with-static-parameters"), lam, g.spnames...)
    end
end

function NamedTuple()
    NamedTuple{(), Tuple{}}(())
end

"""
    NamedTuple{names}(args::Tuple)

Construct a named tuple with the given `names` (a tuple of Symbols) from a tuple of values.
"""
NamedTuple{names}(args::Tuple) where {names} = NamedTuple{names,typeof(args)}(args)

using .Intrinsics: sle_int, add_int

eval(Core, :(NamedTuple{names,T}(args::T) where {names, T <: Tuple} =
             $(Expr(:splatnew, :(NamedTuple{names,T}), :args))))

# constructors for built-in types

import .Intrinsics: eq_int, trunc_int, lshr_int, sub_int, shl_int, bitcast, sext_int, zext_int, and_int

function throw_inexacterror(f::Symbol, @nospecialize(T), val)
    @_noinline_meta
    throw(InexactError(f, T, val))
end

function is_top_bit_set(x)
    @_inline_meta
    eq_int(trunc_int(UInt8, lshr_int(x, sub_int(shl_int(sizeof(x), 3), 1))), trunc_int(UInt8, 1))
end

function is_top_bit_set(x::Union{Int8,UInt8})
    @_inline_meta
    eq_int(lshr_int(x, 7), trunc_int(typeof(x), 1))
end

function check_top_bit(x)
    @_inline_meta
    is_top_bit_set(x) && throw_inexacterror(:check_top_bit, typeof(x), x)
    x
end

function checked_trunc_sint(::Type{To}, x::From) where {To,From}
    @_inline_meta
    y = trunc_int(To, x)
    back = sext_int(From, y)
    eq_int(x, back) || throw_inexacterror(:trunc, To, x)
    y
end

function checked_trunc_uint(::Type{To}, x::From) where {To,From}
    @_inline_meta
    y = trunc_int(To, x)
    back = zext_int(From, y)
    eq_int(x, back) || throw_inexacterror(:trunc, To, x)
    y
end

function toInt8(x::Int8)
    x
end
function toInt8(x::Int16)
    checked_trunc_sint(Int8, x)
end
function toInt8(x::Int32)
    checked_trunc_sint(Int8, x)
end
function toInt8(x::Int64)
    checked_trunc_sint(Int8, x)
end
function toInt8(x::Int128)
    checked_trunc_sint(Int8, x)
end
function toInt8(x::UInt8)
    bitcast(Int8, check_top_bit(x))
end
function toInt8(x::UInt16)
    checked_trunc_sint(Int8, check_top_bit(x))
end
function toInt8(x::UInt32)
    checked_trunc_sint(Int8, check_top_bit(x))
end
function toInt8(x::UInt64)
    checked_trunc_sint(Int8, check_top_bit(x))
end
function toInt8(x::UInt128)
    checked_trunc_sint(Int8, check_top_bit(x))
end
function toInt8(x::Bool)
    and_int(bitcast(Int8, x), Int8(1))
end
function toInt16(x::Int8)
    sext_int(Int16, x)
end
function toInt16(x::Int16)
    x
end
function toInt16(x::Int32)
    checked_trunc_sint(Int16, x)
end
function toInt16(x::Int64)
    checked_trunc_sint(Int16, x)
end
function toInt16(x::Int128)
    checked_trunc_sint(Int16, x)
end
function toInt16(x::UInt8)
    zext_int(Int16, x)
end
function toInt16(x::UInt16)
    bitcast(Int16, check_top_bit(x))
end
function toInt16(x::UInt32)
    checked_trunc_sint(Int16, check_top_bit(x))
end
function toInt16(x::UInt64)
    checked_trunc_sint(Int16, check_top_bit(x))
end
function toInt16(x::UInt128)
    checked_trunc_sint(Int16, check_top_bit(x))
end
function toInt16(x::Bool)
    and_int(zext_int(Int16, x), Int16(1))
end
function toInt32(x::Int8)
    sext_int(Int32, x)
end
function toInt32(x::Int16)
    sext_int(Int32, x)
end
function toInt32(x::Int32)
    x
end
function toInt32(x::Int64)
    checked_trunc_sint(Int32, x)
end
function toInt32(x::Int128)
    checked_trunc_sint(Int32, x)
end
function toInt32(x::UInt8)
    zext_int(Int32, x)
end
function toInt32(x::UInt16)
    zext_int(Int32, x)
end
function toInt32(x::UInt32)
    bitcast(Int32, check_top_bit(x))
end
function toInt32(x::UInt64)
    checked_trunc_sint(Int32, check_top_bit(x))
end
function toInt32(x::UInt128)
    checked_trunc_sint(Int32, check_top_bit(x))
end
function toInt32(x::Bool)
    and_int(zext_int(Int32, x), Int32(1))
end
function toInt64(x::Int8)
    sext_int(Int64, x)
end
function toInt64(x::Int16)
    sext_int(Int64, x)
end
function toInt64(x::Int32)
    sext_int(Int64, x)
end
function toInt64(x::Int64)
    x
end
function toInt64(x::Int128)
    checked_trunc_sint(Int64, x)
end
function toInt64(x::UInt8)
    zext_int(Int64, x)
end
function toInt64(x::UInt16)
    zext_int(Int64, x)
end
function toInt64(x::UInt32)
    zext_int(Int64, x)
end
function toInt64(x::UInt64)
    bitcast(Int64, check_top_bit(x))
end
function toInt64(x::UInt128)
    checked_trunc_sint(Int64, check_top_bit(x))
end
function toInt64(x::Bool)
    and_int(zext_int(Int64, x), Int64(1))
end
function toInt128(x::Int8)
    sext_int(Int128, x)
end
function toInt128(x::Int16)
    sext_int(Int128, x)
end
function toInt128(x::Int32)
    sext_int(Int128, x)
end
function toInt128(x::Int64)
    sext_int(Int128, x)
end
function toInt128(x::Int128)
    x
end
function toInt128(x::UInt8)
    zext_int(Int128, x)
end
function toInt128(x::UInt16)
    zext_int(Int128, x)
end
function toInt128(x::UInt32)
    zext_int(Int128, x)
end
function toInt128(x::UInt64)
    zext_int(Int128, x)
end
function toInt128(x::UInt128)
    bitcast(Int128, check_top_bit(x))
end
function toInt128(x::Bool)
    and_int(zext_int(Int128, x), Int128(1))
end
function toUInt8(x::Int8)
    bitcast(UInt8, check_top_bit(x))
end
function toUInt8(x::Int16)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::Int32)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::Int64)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::Int128)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::UInt8)
    x
end
function toUInt8(x::UInt16)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::UInt32)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::UInt64)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::UInt128)
    checked_trunc_uint(UInt8, x)
end
function toUInt8(x::Bool)
    and_int(bitcast(UInt8, x), UInt8(1))
end
function toUInt16(x::Int8)
    sext_int(UInt16, check_top_bit(x))
end
function toUInt16(x::Int16)
    bitcast(UInt16, check_top_bit(x))
end
function toUInt16(x::Int32)
    checked_trunc_uint(UInt16, x)
end
function toUInt16(x::Int64)
    checked_trunc_uint(UInt16, x)
end
function toUInt16(x::Int128)
    checked_trunc_uint(UInt16, x)
end
function toUInt16(x::UInt8)
    zext_int(UInt16, x)
end
function toUInt16(x::UInt16)
    x
end
function toUInt16(x::UInt32)
    checked_trunc_uint(UInt16, x)
end
function toUInt16(x::UInt64)
    checked_trunc_uint(UInt16, x)
end
function toUInt16(x::UInt128)
    checked_trunc_uint(UInt16, x)
end
function toUInt16(x::Bool)
    and_int(zext_int(UInt16, x), UInt16(1))
end
function toUInt32(x::Int8)
    sext_int(UInt32, check_top_bit(x))
end
function toUInt32(x::Int16)
    sext_int(UInt32, check_top_bit(x))
end
function toUInt32(x::Int32)
    bitcast(UInt32, check_top_bit(x))
end
function toUInt32(x::Int64)
    checked_trunc_uint(UInt32, x)
end
function toUInt32(x::Int128)
    checked_trunc_uint(UInt32, x)
end
function toUInt32(x::UInt8)
    zext_int(UInt32, x)
end
function toUInt32(x::UInt16)
    zext_int(UInt32, x)
end
function toUInt32(x::UInt32)
    x
end
function toUInt32(x::UInt64)
    checked_trunc_uint(UInt32, x)
end
function toUInt32(x::UInt128)
    checked_trunc_uint(UInt32, x)
end
function toUInt32(x::Bool)
    and_int(zext_int(UInt32, x), UInt32(1))
end
function toUInt64(x::Int8)
    sext_int(UInt64, check_top_bit(x))
end
function toUInt64(x::Int16)
    sext_int(UInt64, check_top_bit(x))
end
function toUInt64(x::Int32)
    sext_int(UInt64, check_top_bit(x))
end
function toUInt64(x::Int64)
    bitcast(UInt64, check_top_bit(x))
end
function toUInt64(x::Int128)
    checked_trunc_uint(UInt64, x)
end
function toUInt64(x::UInt8)
    zext_int(UInt64, x)
end
function toUInt64(x::UInt16)
    zext_int(UInt64, x)
end
function toUInt64(x::UInt32)
    zext_int(UInt64, x)
end
function toUInt64(x::UInt64)
    x
end
function toUInt64(x::UInt128)
    checked_trunc_uint(UInt64, x)
end
function toUInt64(x::Bool)
    and_int(zext_int(UInt64, x), UInt64(1))
end
function toUInt128(x::Int8)
    sext_int(UInt128, check_top_bit(x))
end
function toUInt128(x::Int16)
    sext_int(UInt128, check_top_bit(x))
end
function toUInt128(x::Int32)
    sext_int(UInt128, check_top_bit(x))
end
function toUInt128(x::Int64)
    sext_int(UInt128, check_top_bit(x))
end
function toUInt128(x::Int128)
    bitcast(UInt128, check_top_bit(x))
end
function toUInt128(x::UInt8)
    zext_int(UInt128, x)
end
function toUInt128(x::UInt16)
    zext_int(UInt128, x)
end
function toUInt128(x::UInt32)
    zext_int(UInt128, x)
end
function toUInt128(x::UInt64)
    zext_int(UInt128, x)
end
function toUInt128(x::UInt128)
    x
end
function toUInt128(x::Bool)
    and_int(zext_int(UInt128, x), UInt128(1))
end

# TODO: this is here to work around the 4 method limit in inference (#23210).
const BuiltinInts = Union{Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8, Bool}
function Int8(x::BuiltinInts)
    toInt8(x)::Int8
end
function Int16(x::BuiltinInts)
    toInt16(x)::Int16
end
function Int32(x::BuiltinInts)
    toInt32(x)::Int32
end
function Int64(x::BuiltinInts)
    toInt64(x)::Int64
end
function Int128(x::BuiltinInts)
    toInt128(x)::Int128
end
function UInt8(x::BuiltinInts)
    toUInt8(x)::UInt8
end
function UInt16(x::BuiltinInts)
    toUInt16(x)::UInt16
end
function UInt32(x::BuiltinInts)
    toUInt32(x)::UInt32
end
function UInt64(x::BuiltinInts)
    toUInt64(x)::UInt64
end
function UInt128(x::BuiltinInts)
    toUInt128(x)::UInt128
end

function (::Type{T})(x::T) where T <: Number
    x
end

function Int(x::Ptr)
    bitcast(Int, x)
end
function UInt(x::Ptr)
    bitcast(UInt, x)
end
if Int === Int32
Int64(x::Ptr) = Int64(UInt32(x))
UInt64(x::Ptr) = UInt64(UInt32(x))
end
function Ptr{T}(x::Union{Int, UInt, Ptr}) where T
    bitcast(Ptr{T}, x)
end
function Ptr{T}() where T
    Ptr{T}(0)
end

function Signed(x::UInt8)
    Int8(x)
end
function Unsigned(x::Int8)
    UInt8(x)
end
function Signed(x::UInt16)
    Int16(x)
end
function Unsigned(x::Int16)
    UInt16(x)
end
function Signed(x::UInt32)
    Int32(x)
end
function Unsigned(x::Int32)
    UInt32(x)
end
function Signed(x::UInt64)
    Int64(x)
end
function Unsigned(x::Int64)
    UInt64(x)
end
function Signed(x::UInt128)
    Int128(x)
end
function Unsigned(x::Int128)
    UInt128(x)
end

function Signed(x::Union{Float32, Float64, Bool})
    Int(x)
end
function Unsigned(x::Union{Float32, Float64, Bool})
    UInt(x)
end

function Integer(x::Integer)
    x
end
function Integer(x::Union{Float32, Float64})
    Int(x)
end

ccall(:jl_set_istopmod, Cvoid, (Any, Bool), Core, true)
