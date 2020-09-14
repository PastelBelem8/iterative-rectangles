# -----------------------------------------------------------------------
#                                  Tasks
# -----------------------------------------------------------------------
# Randomly generate rectangles
# Draw rectangles
# Read input from user  (if it likes it or not)
# Draw rectangles similar to the ones liked by the user
#  -----------------------------------------------------------------------

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
# - texture: Enum, the texture of the rectangle (to add)

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

# Implementation detail Note:
# ---------------------------
# Since the color elements do not have an ordinal relation amongst them,
# that is red is not less than nor greater than blue, we opted for not
# indexing the feature `color`. Instead, we leave it to downstream tasks
# which will exhibit specificities regarding the inputs they accept
# (e.g., explore the semantic meaning of each color).
as_array(r::Rectangle) = [r.x r.y r.width r.height r.color]

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

# Type alias
const Literal = Union{String,Real}

abstract type Indexer end

cardinality(indexer::Indexer) = length(values(ix.values))
output_columns(indexer::Indexer) =
    map((v) -> "$(ix.feature_id)_$v", keys(ix.values))

function transform(indexer::Indexer, feature) end


mutable struct OneHotIndexer <: Indexer
    feature_id::Literal
    values::Dict{Literal,String}
end

OneHotIndexer(feature_id::Literal, feature) =
    let u_values = unique(feature),
        u_ids = 1:length(unique_values),
        values = map((v, id) -> v => id, u_values, u_ids)

        OneHotIndexer(feature_id, Dict(values))
    end

# TODO -> Values in data not present in indexer.values?
#      -> persist & load?
transform(indexer::OneHotIndexer, feature) =
    let n_rows = length(feature),
        cardinality = length(indexer.values),
        one_hot_matrix = zeros(Int8, n_rows, cardinality),
        one_hot_values = map(f -> get(indexer.values, f), feature)

        for (row_ix, row_col) in zip(1:n_rows, one_hot_values)
            one_hot_matrix[row_ix, row_col] = 1
        end
        one_hot_matrix
    end


mutable struct LabelIndexer <: Indexer
    feature_id::Literal
    values::Dict{Literal,Int}
end

LabelIndexer(feature_id::Literal, feature) =
    let u_values = unique(feature),
        u_ids = 1:length(unique_values),
        values = map((v, id) -> v => id, u_values, u_ids)

        print(feature_id, Dict(values))
        LabelIndexer(feature_id, Dict(values))
    end

transform(indexer::LabelIndexer, feature) =
    map(f -> get(indexer.values, f), feature)

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
mutable struct Dataset
    X::Matrix{Any}
    y::Matrix{Int64}
    categoricals::Vector{Int64}
end

Dataset(n_features, n_labels, categoricals) =
    let X = Array{Any,2}(undef, 0, n_features),
        y = Array{Int64,2}(undef, 0, n_labels)

        Dataset(X, y, categoricals)
    end

n_features(data::Dataset) = size(data.X, 2)
n_rows(data::Dataset) = isempty(data) ? 0 : size(data.X, 1)

categoricals(data::Dataset) = data.categoricals
numericals(data::Dataset) = setdiff(1:n_features(data), categoricals(data))

get_data(data::Dataset) = data.X
get_label(data::Dataset) = data.y
get_feature(name, data::Dataset) = data.X[:, name]
get_features_names(data::Dataset, except = nothing) =
    except === nothing ? Set(1:n_features(data)) :
    setdiff(Set(1:n_features(data)), Set(except))

import Base: push!, isempty

Base.push!(data::Dataset, obj, label) =
    let features = as_array(obj)
        data.X = [data.X; features]
        data.y = [data.y; label]
    end

Base.isempty(data::Dataset) = isempty(data.X) && isempty(data.y)

filter_by_y(data::Dataset, predicate) =
    if isempty(data)
        data
    else
        let X = data.X,
            y = data.y,
            n_instances = size(X, 1),
            ixs = filter((i) -> predicate(y[i, :]), 1:n_instances)

            Dataset(X[ixs, :], y[ixs, :], categoricals(data))
        end
    end

using DelimitedFiles
filename = Parameter("data.csv")

write(dataset::Dataset) =
    let X = dataset.X, y = dataset.y
        open(filename(), "w") do io
            writedlm(io, [X y], ',')
        end
    end

transform(indexer::T, feature_id, Xs...) where T =
    let all_features = size(Xs[1], 2),
        features = map(X -> X[:, feature_id], Xs),
        indexer = indexer[1],
        indexer = indexer(feature_id, features[1]) #=,
        transf_features = map((f) -> transform(indexer, f), features),
        non_transf_features = setdiff(Set(1:all_features), feature_id),
        non_transf_features = collect(non_transf_features),

        transf_matrices = map(X -> X[:, non_transf_features], Xs),
        transf_matrices =
            map((M, T) -> [M T], transf_matrices, transf_features)

        transf_matrices
=#
    end


transform(indexers::Vector{T}, features_id, Xs) where T = foldl(
    (Xs, (ix, f_id)) -> transform(ix, f_id, Xs),
    zip(indexers, features_id),
    Xs,
)

#
#   Main Loop
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
