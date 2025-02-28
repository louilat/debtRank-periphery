using CSV
using DataFrames

include("src/inputs/debtrank_initialization.jl")
include("src/processing/iterations.jl")

capitals = CSV.read("sample_data/capitals.csv", DataFrame)
economic_values = CSV.read("sample_data/economic_values.csv", DataFrame)
investments = CSV.read("sample_data/investments.csv", DataFrame)

shock = 0.1
impacted_node = "USD Coin"

model = DRLstm(investments, capitals, economic_values)

run_model(model, 0.2, "Wrapped Ether", 15)