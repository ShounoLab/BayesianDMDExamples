using StatsBase
using Plots
using JLD2
using Random
using DataFrames
using CSV

path_BDMDVMF = "$(@__DIR__)/../BDMD-VMF"
path_TBDMD = "$(@__DIR__)/../TakeishiBayesianDMD"

include("$path_TBDMD/julia/bayesianDMD.jl")
include("$path_BDMDVMF/julia/variationalsvd.jl")
include("$path_BDMDVMF/julia/bayesiandmd.jl")
include("$(@__DIR__)/../julia/DMD.jl")
include("$(@__DIR__)/../NonlinearSimulations/NLSE/pseudospectral.jl")


outdir = "$(@__DIR__)/../output"

if !isdir(outdir)
    mkdir(outdir)
end


### Nonlinear Schrodinger Equation
Ngrids = 256 # Number of Fourier modes
L = 30.0 # Space period
Δt = 2π / 256
t_end = 2π
Nsteps = round(Int, t_end / Δt)

config = NLSESettings(Nsteps, Δt, t_end, Ngrids, L)

# Initial state of the wave function
ψ₀ = 2.0 * sech.(config.gridpoints)

# Generate NLSE data
result = PseudoSpectral(ψ₀, config)
X = result.ψ

# Input dimension and sample size
D, T = size(result.ψ)

# Timepoints
t_ary = collect(0:config.Δt:config.t_end)[1:end - 1]

K = 4 # Number of modes

### Standard DMD ###
naive_dp = solve_dmd(X, K)
X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)


### Takeishi's Bayesian DMD ###
Random.seed!(123)
tmc_config = TMCMCConfig(7500, 5000, thinning = 1)

# Give weakly informative priors
model_params = ModelParams(T, D, K, 1e-3, 1e-3, 1e-3, 1e-3)

# Run Gibbs sampler
tbdmd_ary, tbdmd_logliks = bayesianDMD(X, model_params, tmc_config)

# Check convergence
plot(tbdmd_logliks)

# EAP estimation
tbdmd_mean = mean_bdmd(tbdmd_ary, model_params)
X_pointest_tbdmd = reconstruct_pointest(tbdmd_mean, model_params)

# Save result
@save "$outdir/mcmc_nlse_takeishi.jld2" X X_pointest_tbdmd tmc_config model_params tbdmd_ary tbdmd_logliks


### BDMD-VMF ###
Random.seed!(123)

# Apply variational matrix factorization
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X, K, 200, σ²_U = 1 / D, svdinit = true,
                                                 learn_C_V = true)

# MCMC configuration
mc = MCMCConfig(7500, 5000, 1e-2, 1e-1, 1e-2)

# Give weakly informative priors
hp = BDMDHyperParams(sp.Ubar, sp.Σbar_U, 1e5, 1e5, 0.01, 0.01, D, T, K)

# Run Metropolis sampler
bdmdvmf_ary, bdmdvmf_logliks = run_sampling(X, hp, mc)
# Check convergence
plot(bdmdvmf_logliks)

# EAP estimation
bdmdvmf_mean = mean_bdmd(bdmdvmf_ary, hp, mc)
X_pointest_bdmdvmf = reconstruct_pointest(bdmdvmf_mean, hp)

# Save result
@save "$outdir/mcmc_nlse_bdmdvmf.jld2" X X_pointest_bdmdvmf bdmdvmf_ary bdmdvmf_logliks hp mc


### Visualize results
X1, X2, X3, X4 = abs.(X), abs.(X_reconst_dmd), abs.(X_pointest_tbdmd), abs.(X_pointest_bdmdvmf)
cmin = minimum(hcat(X1, X2, X3, X4))
cmax = maximum(hcat(X1, X2, X3, X4))
p1 = heatmap(t_ary, config.gridpoints, X1,
             title = "original",
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p2 = heatmap(t_ary, config.gridpoints, X2,
             title = "DMD",
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p3 = heatmap(t_ary, config.gridpoints, X3,
             title = "Takeishi's Bayesian DMD",
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p4 = heatmap(t_ary, config.gridpoints, X4,
             title = "BDMD-VMF",
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p = plot(p1, p2, p3, p4)
savefig(p, "$outdir/NLSE_reconst.pdf")


### Compute RMSEs
RMSEs = DataFrame(Method = ["DMD", "TakeishiBDMD", "BDMD-VMF"],
                  NLSE = fill(0.0, 3))

RMSEs[:NLSE][1] = √mean(abs2.(X - X_reconst_dmd))
RMSEs[:NLSE][2] = √mean(abs2.(X - X_pointest_tbdmd))
RMSEs[:NLSE][3] = √mean(abs2.(X - X_pointest_bdmdvmf))

CSV.write("$outdir/RMSEs_NLSE.csv", RMSEs)

