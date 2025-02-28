"""DebtRank LSTM Type definition"""

struct DRLstm
    nodes_id::DataFrame
    nodes_values::Vector
    W::Matrix
    # M::Matrix
end

function (layer::DRLstm)((c, h)::Tuple{Vector, Vector})
    # p = (1 .- c) .* sigmoid_fast.(100 .* (h .- 0.05))
    # p = (1 .- c) .* max.(0, min.(1, 20 .* h))
    p = (1 .- c) .* Int.(h .> 0)
    # println(p)
    next_c = c .+ p
    next_h = cmin.(h + transpose(layer.W) * (h .* p))
    return next_c, next_h
end

function (layer::DRLstm)((c, h, M)::Tuple{Vector, Vector, Matrix})
    # p = (1 .- c) .* sigmoid_fast.(100 .* (h .- 0.05))
    # p = (1 .- c) .* max.(0, min.(1, 20 .* h))
    p = (1 .- c) .* Int.(h .> 0)
    # println(p)
    next_c = c .+ p
    next_h = cmin.(h + transpose(layer.W .* M) * (h .* p))
    return next_c, next_h, M
end

function cmin(x)
    return .5 * (1 + x - sqrt((1 - x)^2 + 1e-4))
end