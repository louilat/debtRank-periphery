using CSV
using DataFrames

include("src/inputs/debtrank_initialization.jl")
include("src/processing/iterations.jl")
include("src/outputs/generate_outputs.jl")

capitals = CSV.read("sample_data/capitals.csv", DataFrame)
economic_values = CSV.read("sample_data/economic_values.csv", DataFrame)
investments = CSV.read("sample_data/investments.csv", DataFrame)

# capitals = DataFrame(
#     node = ["A", "B", "C", "d", "e"],
#     capital = [10, 10, 10, 1, 1],
# )

# investments = DataFrame(
#     target=["A", "A", "d", "e"],
#     source=["d", "e", "B", "C"],
#     amount = [3, 5, 2, 4],
# )

# economic_values = DataFrame(
#     node = ["B", "A", "C", "d", "e"],
#     weight = [0.4, 0.2, 0.4, 0, 0]
# )

shock = 0.1
impacted_node = "USD Coin"

model = DRLstm(investments, capitals, economic_values)

output = run_model(model, 0.2, impacted_node, 15).W

output_data = generate_marginal_effects(output)

CSV.write("marginal_effects.csv", output_data)
