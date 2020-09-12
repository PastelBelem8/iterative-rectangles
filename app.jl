# -----------------------------------------------------------------------
#                                  Tasks
# -----------------------------------------------------------------------
# Randomly generate rectangles
# Draw rectangles
# Read input from user  (if it likes it or not)
# Draw rectangles similar to the ones liked by the user
#     - EMA (weight different models based on the most recent)
#     - Clustering (after achieving rectangles above a certain amount
#        of rectangles) - how to determine number of clusters??
#     - ML to generate around the clusters the user likes the most.
#  -----------------------------------------------------------------------

#=
    This file is the identical copy to the one found at
    https://github.com/aptmcl/Khepri.jl/blob/master/src/Parameters.jl
    The repository in question is under constant changes and, therefore,
    we opted for having a local copy


    An example of usage is
    xpto = Parameter(1)
    foo(x) = x + xpto()
    bar(y) = with(xpto, 3) do
        foo(y)
    end
    foo(2)
    bar(2)
    xpto()
=#

mutable struct Parameter{T}
  value::T
end

(p::Parameter)() = p.value
(p::Parameter)(newvalue) = p.value = newvalue

function with(f, p, newvalue)
  oldvalue = p()
  p(newvalue)
  try
    f()
  finally
    p(oldvalue)
  end
end

with(f, p, newvalue, others...) =
  with(p, newvalue) do
    with(f, others...)
  end

# min max values for width and height parameters of the rectangle
MIN, MAX = Parameter(0.1), Parameter(100)

# Task 1. Randomly generate rectangles
# Each rectangle is generated based on 4 different inputs:
# - x: real, the x coordinate of the upper left corner of the rectangle
# - y: real, the y coordinate of the upper left corner of the rectangle
# - width: real, the width of the rectangle
# - height: real, the height of the rectangle
# - color: string, the name of the color of the rectangle
@enum Color begin
    black  = 1
    blue   = 2
    red    = 3
    orange = 4
    yellow = 5
    green  = 6
    purple = 7
    brown  = 8
    pink   = 9
end

struct Rectangle
    x::Real
    y::Real
    width::Real
    height::Real
    color::Color
    Rectangle(x, y, width, height, color) =
      width <= 0 || height <= 0 ? error("width and height must be positive") : new(x, y, width, height, color)
end

Rectangle() = Rectangle(0, 0)
Rectangle(x, y) = Rectangle(x, y, 1, 1, black)

# Note: Since the color elements do not have an ordinal relation amongst
# them, I've decided not to transform the color into an integer and,
# instead, leave that transformation/indexing for the downstream task.
# Some algorithms might explore the semantic existing in the name of
# the colors.
as_array(r::Rectangle) = [r.x, r.y, r.width, r.height, r.color]

using Random
import Random.rand

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Rectangle}) =
    let vals = rand(2),
        vals = MIN() .+ vals .* (MAX() - MIN())
        vals = round.(vals, digits=2)
        color = rand(instances(Color), 1)[1]

        Rectangle(0, 0, vals..., color)
    end

# Task 2. Loop
# - Generate random rectangle
# - Read user's input
# - Associate it to rectangle
# - Store

# Task 2.1. Store rectangles, label / user_input
# The goal is to have a way of representing the properties of the rectangles
# (also called features) in a matrix format and associate it to the user's
# input (also called the label).
#
# The properties of the Rectangles will be represented in a column format
#  <x, y, width, height, color> and it will be associated to a label <like_it>
#
# To achieve this, we're creating a specialization of the generic method
# push! which receives a rectangle and adds its properties to the specified
# matrix.
import Base: push!

push!(X::Array{T, 1}, rect::Rectangle) where T =
    let features = as_array(rect)
        push!(X, features)
    end

# Task 2.2. Persist it



using Random
Random.seed!(1234567)

let user_input = "",
    X = Array{Any, 1}[],
    y = Vector{Int64}()
    while true
        println("Generating rectangle...")
        rectangle = rand(Rectangle)
        println(rectangle)
        print("Do you like it? (1=yes, 0=no)\n > ")
        user_input = readline(stdin)

        if user_input == "quit" break end
        user_input = parse(Int, user_input)
        push!(X, rectangle)
        push!(y, user_input)
    end
end
