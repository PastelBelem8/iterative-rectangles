
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
