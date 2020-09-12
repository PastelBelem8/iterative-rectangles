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
Rectangle(x, y) = Rectangle(x, y, 1, 1, )


using Random
Random.seed!(1234567)

# min max values for width and height parameters of the rectangle
MIN, MAX = Parameter(0.1), Parameter(100)

import Random.rand

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Rectangle}) =
    let vals = rand(2),
        vals = MIN() .+ vals .* (MAX() - MIN())
        vals = round.(vals, digits=2)
        color = rand(instances(Color), 1)[1]

        Rectangle(0, 0, vals..., color)
    end

rand(Rectangle)



# Task 2. Loop
# - Generate random rectangle
# - Read user's input
# - Associate it to rectangle
# - Store

macro until(condition, expression)
    quote
        while !($condition)
            $expression
        end
    end |> esc
end





let user_input = ""
    @until user_input == "quit" begin
        println("Generating rectangle...")
        rectangle = rand(Rectangle)
        println(rectangle)
        print("Do you like it? (1=yes, 0=no)\n > ")
        user_input = parse(Int, readline(stdin))
    end
end
