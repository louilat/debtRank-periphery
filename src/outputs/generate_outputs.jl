"""Functions to generate outputs files"""

using DataFrames

function generate_marginal_effects(∂S∂W::Matrix)::DataFrame
    output = DataFrame(
        source=[],
        target=[],
        marginal_effect=[],
    )
    N = size(∂S∂W)[1]
    for i in 1:N
        effect_i = DataFrame(
            source=i,
            target=1:N,
            marginal_effect=∂S∂W[i, :]
        )
        effect_i = effect_i[effect_i.marginal_effect .> 0]
        output = vcat(output, effect_i)
    end
    return output
end