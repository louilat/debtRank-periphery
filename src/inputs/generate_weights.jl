"""Functions for initializing DebtRank from inputs"""

using DataFrames
using SparseArrays

function encode_input_nodes(
    investments::DataFrame,
    capitals::DataFrame,
    economic_values::DataFrame
)::Tuple{DataFrame, DataFrame, DataFrame, DataFrame}

    nodes_id::DataFrame = select(capitals, :node => :node)
    nodes_id.id = 1:nrow(nodes_id)
    capitals_::DataFrame = leftjoin(capitals, nodes_id, on = :node)
    select!(capitals_, :id => :node, :capital => :capital)

    investments_::DataFrame = leftjoin(investments, nodes_id, on = :source => :node)
    select!(investments_, :id => :source, :target => :target, :amount => :amount)

    investments_ = leftjoin(investments_, nodes_id, on = :target => :node)
    select!(investments_, :source => :source, :id => :target, :amount => :amount)

    economic_values_::DataFrame = leftjoin(economic_values, nodes_id, on = :node)
    select!(economic_values_, :id => :node, :weight => :weight)

    return nodes_id, investments_, capitals_, economic_values_
end

function generate_weights_matrix(investments::DataFrame, capitals::DataFrame)::SparseMatrixCSC
    investments_::DataFrame = leftjoin(investments, capitals, on = :source => :node)
    select!(investments_, :source => :source, :target => :target, [:amount, :capital] => ((a, c) -> a ./ c) => :weight)
    W = sparse(investments_.target, investments_.source, investments_.weight)
    return min.(1, W)
end

function generate_bweights_matrix(investments::DataFrame, capitals::DataFrame)::SparseMatrixCSC
    investments_::DataFrame = leftjoin(investments, capitals, on = :target => :node)
    select!(investments_, :source => :source, :target => :target, [:amount, :capital] => ((a, c) -> a ./ c) => :weight)
    W = sparse(investments_.source, investments_.target, investments_.weight)
    return min.(1, W)
end

function generate_connection_matrix(economic_values::DataFrame)::SparseMatrixCSC
    sort!(economic_values, :node)
    is_reserve::Vector{Int} = economic_values.weight .> 0
    M::Matrix{Int} = is_reserve * transpose(1 .- is_reserve) + (1 .- is_reserve) * transpose(is_reserve)
    return sparse(M)
end