
using DataFrames
using SparseArrays

include("generate_weights.jl")
include("../types.jl")

function DRLstm(investments::DataFrame, capitals::DataFrame, economic_values::DataFrame)::DRLstm
    nodes_id, investments_, capitals_, economic_values_ = encode_input_nodes(investments, capitals, economic_values)
    W::SparseMatrixCSC = generate_weights_matrix(investments_, capitals_)
    # M::SparseMatrixCSC = generate_connection_matrix(economic_values_)
    sort!(economic_values, :node)
    return DRLstm(nodes_id, economic_values.weight, W)
end