### handle HuGaDB Dataset
### see details in: https://github.com/romanchereshnev/HuGaDB
### Chereshnev R., Kertész-Farkas A. (2018) HuGaDB: Human Gait Database for Activity Recognition
### from Wearable Inertial Sensor Networks. In: van der Aalst W. et al. (eds)
### Analysis of Images, Social Networks and Texts. AIST 2017.
### Lecture Notes in Computer Science, vol 10716. Springer, Cham
### https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12


using Plots
using DelimitedFiles
using CSV
using DataFrames
using JLD2
using Random

path_BDMDVMF = "$(@__DIR__)/../BDMD-VMF"

include("$path_BDMDVMF/julia/variationalsvd_missingvals.jl")
include("$path_BDMDVMF/julia/bayesiandmd_missingvals.jl")

outdir = "$(@__DIR__)/../output"

if !isdir(outdir)
    mkdir(outdir)
end

# Load HuGaDB file
fname = "$(@__DIR__)/../data/HuGaDB/HuGaDB_v1_bicycling_01_01.txt" df = CSV.read(fname, delim = '\t', header = 4)

# Extract specific interval
datarange = collect(901:1050)
X = Matrix(Matrix{Union{Missing, Complex{Float64}}}(df[datarange, r"g"])')

# Linear transformation into [-1, 1]
minx, maxx = minimum(real.(X)), maximum(real.(X))
X = real.(2 / (maxx - minx) .* (X .- minx) .- 1)
X = Matrix{Union{Missing, ComplexF64}}(X)

# Make masked data
X_missing = deepcopy(X)
X_missing[9:end, 76:end] .= missing
X_missing[1:12, 1:60] .= missing

# Save input matrix
writedlm("$outdir/hugadb_original.csv", real.(X), ",")
writedlm("$outdir/hugadb_missing.csv", real.(X_missing), ",")


# Number of modes
K = 2

# Input dimension and sample size
D, T = size(X)

### BDMD-VMF ###
Random.seed!(123)


# Apply variational matrix factorization
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X, K, 200,
                                                 σ²_U = 1e10,
                                                 σ²_V = 1.0,
                                                 learn_C_V = false)

# MCMC configuration
mc = MCMCConfig(7500, 5000, 1e-2, 1e-1, 1e-2)

# Give weakly informative priors
hp = BDMDHyperParams(sp, vhp) # Run Metropolis sampler
bdmdvmf_ary, logliks = run_sampling(X_missing, hp, mc)

# Check convergence
plot(logliks)

# Interval and point reconstruction
X_preds = reconstruct_bdmd_missingvals(bdmdvmf_ary, hp, sp, mc.n_iter, mc.burnin)
@save "$outdir/mcmc_hugadb_bdmdvmf.jld2" X X_missing X_preds bdmdvmf_ary hp sp vhp logliks


# Load MCMC result
#@load "$outdir/mcmc_hugadb_bdmdvmf.jld2" X X_missing X_preds bdmdvmf_ary hp sp vhp logliks

# Load VAR(2) result
df_stan_0025 = CSV.read("$outdir/df_0025.csv")[:, 2:end]
df_stan_mean = CSV.read("$outdir/df_mean.csv")[:, 2:end]
df_stan_0975 = CSV.read("$outdir/df_0975.csv")[:, 2:end]

# Make matrix for quantiles
X_quantiles_real, X_quantiles_imag = get_quantiles(X_preds, interval = 0.95)
X_quantiles_stan = Array{Float64}(undef, size(X_quantiles_real))
X_quantiles_stan[:, :, 1] .= Matrix(df_stan_0025)
X_quantiles_stan[:, :, 2] .= Matrix(df_stan_0975)


# Plot
for d in [5, 10, 14, 18]
    p = plot(real.(X[d, :]), dpi = 300, ribbon = (real.(X[d, :]) .- X_quantiles_stan[d, :, 1],
                                            X_quantiles_stan[d, :, 2] .- real.(X[d, :])),
            line = (:dot, 4), label = "original", legend = :none,
            linecolor = :deepskyblue4, fillcolor = :darkslategray, fillalpha = 0.3,
            xtickfontsize = 14,
            ytickfontsize = 14)
    plot!(real.(X[d, :]), dpi = 300, ribbon = (real.(X[d, :]) .- X_quantiles_real[d, :, 1],
                                            X_quantiles_real[d, :, 2] .- real.(X[d, :])),
        line = (:dot, 4), label = "original", legend = :none,
        linecolor = :deepskyblue4, fillcolor = :brown, fillalpha = 0.3)
    plot!(real.(X_missing[d, :]), line = (:solid, 3), label = "observed",
        markercolor = :royalblue3, markeralpha = 0.5,
        markersize = 6, markerstrokewidth = 2,
        seriestype = :scatter)
    p = plot(p, dpi = 300)
    savefig(p, "$outdir/cycling_$(string(names(df[:, r"g"])[d]))_95.pdf")
end
