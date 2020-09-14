#=
    The first part of this file is the identical copy to the one found at
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

# ------------------------------------------------------------------------
# Task 1. Create an abstraction - Rectangle
# ------------------------------------------------------------------------
# To assess this idea we are going to create a prototype.
# In the early stages, we will use the **Rectangle** abstraction.
#
# Each Rectangle is characterized by 6 different properties:
# - x: real, the x coordinate of the lower left corner of the rectangle
# - y: real, the y coordinate of the lower left corner of the rectangle
# - width: real, the width of the rectangle
# - height: real, the height of the rectangle
# - color: Enum, the color of the rectangle
@enum Color begin
    black = 1
    blue = 2
    red = 3
    orange = 4
    yellow = 5
    green = 6
    purple = 7
    brown = 8
    pink = 9
end

size_enum(x) = length(instances(x))

struct Rectangle
    x::Real
    y::Real
    width::Real
    height::Real
    color::Color
    Rectangle(x, y, width, height, color) =
        width <= 0 || height <= 0 ? error("width and height must be positive") :
        new(x, y, width, height, color)
end

Rectangle() = Rectangle(0, 0)
Rectangle(x, y) = Rectangle(x, y, 1, 1, black)

n_features(::Type{Rectangle}) = 5
categoricals(::Type{Rectangle}) = [5]

using Random
import Random.rand

MIN = Parameter(0)
MAX = Parameter(1000)

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Rectangle}) =
    let vals = rand(2), vals = MIN() .+ vals .* (MAX() - MIN())
        vals = round.(vals, digits = 2)
        color = rand(instances(Color), 1)[1]

        Rectangle(0, 0, vals..., color)
    end

rand(Rectangle)
rand(Rectangle, 2)


# Implementation detail Note:
# ---------------------------
# Since the color elements do not have an ordinal relation amongst them,
# that is red is not less than nor greater than blue, we opted for not
# indexing the feature `color`. Instead, we leave it to downstream tasks
# which will exhibit specificities regarding the inputs they accept
# (e.g., explore the semantic meaning of each color).
as_array(r::Rectangle) = [r.x r.y r.width r.height r.color]
as_array_label_enc(r::Rectangle) = [r.x r.y r.width r.height Int(r.color)]

# Note: This function assumes the Enum values to be sequential and to start at 1
as_array_one_hot_enc(r::Rectangle) =
    let n_colors = size_enum(Color),
        one_enc = zeros(Int8, n_colors)
        one_enc[Int(r.color)] = 1

        [r.x r.y r.width r.height one_enc...]
    end

#=
r = rand(Rectangle)
as_array(r)
as_array_label_enc(r)
as_array_one_hot_enc(r)
=#

labels(r::Rectangle) = ["x", "y", "width", "height", "color"]
labels_one_hot_enc(r::Rectangle) =
    let color_labels = map(c -> "color_$c", instances(Color)),
        color_labels = collect(color_labels)
        ["x", "y", "width", "height", color_labels...]
    end

#=
labels(r)
labels_one_hot_enc(r)
=#

# ------------------------------------------------------------------------
# Task 2. Create the Read-Eval-Print Loop
# ------------------------------------------------------------------------
# The goal is to have an iterative mechanism that is able to repeat the
# following rationale:
# - Generate an object according to a specified `generate` function and the
#   current dataset;
# - Show the object using the `draw` function and ask for user's input;
# - Parse user's input using `parser` function;
# - Update the current dataset.
# - Read user's input
# - Associate it to rectangle
# - Store

# ------------------------------------------------------------------------
# Task 2.1. Store object and user input
# ------------------------------------------------------------------------
# The goal is to have a way of representing the properties of the rectangles
# (also called features) in a matrix format and associate it to the user's
# input (also called the label).
#
# The properties of the Rectangles will be represented in a column format
#  <x, y, width, height, color> and it will be associated to a label <like_it>
#
# To abstract this rationale from the user, the Dataset abstraction is created.
# Additionally, Base.push! is specialized for a Dataset instance and is able
# to add any object <: Shape to the specified dataset.
abstract type AbstractDataset end

mutable struct Dataset{T} <: AbstractDataset
    X::Array{T, 1}
    y::Array{Int8, 1}
end

Dataset{T}() where T = Dataset{T}(T[], Int8[])

#=
Dataset{Rectangle}()
d = Dataset{Rectangle}()
=#

get_data(d::Dataset, repr=identity) = repr(d.X)
get_label(d::Dataset, repr=identity) = repr(d.y)

filter_by_label(d::Dataset{T}, pred) where T =
    if isempty(d)
        d
    else
        let Xs = get_data(d),
            ys = get_label(d),
            n_rows = length(Xs),
            ixs = filter(ix -> pred(ys[ix]), 1:n_rows)
            Dataset{T}(Xs[ixs], ys[ixs])
        end
    end

#=
rects = [rand(Rectangle), rand(Rectangle)]
d = Dataset{Rectangle}(rects, [0, 1])
filter_by_label(d, y -> y == 1.0)
=#

import Base: push!, isempty

Base.push!(d::AbstractDataset, obj, label) =
    begin
        push!(d.X, obj)
        push!(d.y, label)
    end

Base.isempty(data::Dataset) = isempty(data.X) && isempty(data.y)

#=
d = Dataset{Rectangle}()
isempty(d)
push!(d, rand(Rectangle), 1)
isempty(d)
d.X, d.y
=#

using DelimitedFiles
filename = Parameter("data.csv")

import Base: write

write(d::Dataset) =
    let Xs = get_data(d),
        Xs = map(as_array, Xs),
        Xs = vcat(Xs...),
        ys = get_label(d)
        open(filename(), "w") do io
            writedlm(io, [Xs ys], ',')
        end
    end

#
#
#   Main Loop
#
# 
iterative_rectangles(
    shape,
    generate,
    draw,
    prompt,
    parser,
    seed::Int,
    max_attempts::Int,
) =
    let label = "",
        n_features = n_features(shape),
        data = Dataset(n_features, 1, categoricals(shape))

        for i = 1:max_attempts
            let object = generate(data, i, shape)
                draw(object)
                label = parser(prompt())
                if label === nothing
                    write(data)
                    break
                end
                push!(data, object, label)
            end
        end
    end

# ------------------------------------------------------------------------
# Task 3. Create the Heuristics
# ------------------------------------------------------------------------
# For this task, we need to create the base methods to work with the
# iterative-rectangles method that we defined.
#
# First, we're going to define 3 simple functions:
#     - `cli_draw: Shape`: receives a shape and draws its representation
#        in the cli.
#     - `cli_input`: asks for the user input.
#     - `simple_parser`: attempts to convert input to an integer and
#        returns nothing otherwise.

cli_draw(object) = println(object)
cli_input() = begin
    print("Do you like the shape above? (1=yes, 0=no)\n > ")
    readline(stdin)
end
simple_parser(input::String) = tryparse(Int64, input)

# Second, we're going to define the generators:
# --> Heuristic 1: Random Rectangle
#     Generates random rectangles regardless of the current dataset
generate_random(dataset, iter, object_type) = rand(object_type)



#=
using Random
Random.seed!(1234567)

with(filename, "data_random.csv") do
    Random.seed!(1234567)
    iterative_rectangles(
        Rectangle, generate_random, cli_draw, cli_input, simple_parser, 42, 1000
    )
end
=#
# --> Heuristic 2: Perturbations + Similarity
#     Generates random rectangles returns the one which maximizes the
#     cosine similarity to those already liked by the user.
using Distances

bootstrap_samples = Parameter(1)
n_samples = Parameter(50)
distance_metric = Parameter(CosineDist())
indexers = Parameter(Dict(
    5 => LabelIndexer, # Color (feature 5) will be indexed
))


# Generate samples
# fit_transform dataset
# transform samples (if necessary)
# compute pairwise similarity between dataset and samples
generate_similar(dataset, iter, object_type) =
    if n_rows(dataset) <= bootstrap_samples()
        generate_random(dataset, iter, object_type)
    else
        let n_samples = n_samples(),
            samples = rand(object_type, n_samples),
            S = vcat(map(as_array, samples)...),
            dataset = filter_by_y(dataset, (y) -> y == 1),
            X = get_data(dataset),
            y = get_label(dataset),
            # Indexing
            features_lst = collect(keys(indexers())),
            indexers_lst = map(f -> get(indexers(), f, nothing), features_lst),
            (X_indexed, samples_indexed) = transform(indexers_lst, features_lst, (X, S)...),
            # This yields an sm x s matrix
            dist_metric = distance_metric(),
            distances = pairwise(dist_metric, X_indexed', samples_indexed'),
            distances_sum = mapslices(sum, distances, dims = 2),

            # Pick
            sample_ix = argmax(distances_sum)
            samples[sample_ix]
        end
    end


with(filename, "data_similar.csv") do
    Random.seed!(1234567)
    iterative_rectangles(
        Rectangle,
        generate_similar,
        cli_draw,
        cli_input,
        simple_parser,
        42,
        1000,
    )
end

# There are many
#     - EMA (weight different models based on the most recent)
#     - Clustering (after achieving rectangles above a certain amount
#        of rectangles) - how to determine number of clusters??
#     - ML to generate around the clusters the user likes the most.
