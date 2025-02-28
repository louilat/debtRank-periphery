using Flux

include("../types.jl")

function fwd(model::DRLstm, tau0, h0, n_iter)
    t = tau0
    h = h0
    for _ in 1:n_iter
        t, h = model((t, h))
    end
    s = sum(h .* model.nodes_values)
    # println(h[1:3])
    return s
end

function run_model(debtrank::DRLstm, shock::Float64, impacted_node::String, n_iter::Int)
    impacted_node_id = debtrank.nodes_id[(debtrank.nodes_id.node .== impacted_node), :id][1]
    h0 = zeros(nrows(debtrank.nodes_id))
    h0[impacted_node_id] = shock
    t0 = zeros(nrows(debtrank.nodes_id))

    grad = Flux.gradient(model -> fwd(model, t0, h0, n_iter), debtrank)
    return grad
end