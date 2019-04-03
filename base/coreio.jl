# This file is a part of Julia. License is MIT: https://julialang.org/license

function print(xs...)
    print(stdout::IO, xs...)
end
function println(xs...)
    println(stdout::IO, xs...)
end
function println(io::IO)
    print(io, '\n')
end

function show end
function repr end

struct DevNull <: IO end
const devnull = DevNull()
function isreadable(::DevNull)
    false
end
function iswritable(::DevNull)
    true
end
function isopen(::DevNull)
    true
end
function read(::DevNull, ::Type{UInt8})
    throw(EOFError())
end
function write(::DevNull, ::UInt8)
    1
end
function unsafe_write(::DevNull, ::Ptr{UInt8}, n::UInt)::Int
    n
end
function close(::DevNull)
    nothing
end
function flush(::DevNull)
    nothing
end
function wait_connected(::DevNull)
    nothing
end
function wait_readnb(::DevNull)
    wait()
end
function wait_readbyte(::DevNull)
    wait()
end
function wait_close(::DevNull)
    wait()
end
function eof(::DevNull)
    true
end

let CoreIO = Union{Core.CoreSTDOUT, Core.CoreSTDERR}
    global write, unsafe_write
    write(io::CoreIO, x::UInt8) = Core.write(io, x)
    unsafe_write(io::CoreIO, x::Ptr{UInt8}, nb::UInt) = Core.unsafe_write(io, x, nb)
end

stdin = devnull
stdout = Core.stdout
stderr = Core.stderr
