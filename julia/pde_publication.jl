using StatsBase
using DataFrames
using Plots
using JLD2
using CSV
include("$(@__DIR__)/../julia/DMD.jl")
include("$(@__DIR__)/../NonlinearSimulations/NLSE/pseudospectral.jl")
include("$(@__DIR__)/../NonlinearSimulations/Burgers/cranknicolson.jl")


outdir = "$(@__DIR__)/../output"

RMSEs = DataFrame(Method = ["DMD", "TakeishiBDMD", "BDMD-VMF"],
                  NLSE = fill(0.0, 3),
                  Burgers = fill(0.0, 3))


### NLSE
@load "$outdir/mcmc_nlse_bdmdvmf.jld2" X X_pointest_bdmdvmf
@load "$outdir/mcmc_nlse_takeishi.jld2" X_pointest_tbdmd


Ngrids = 256 # Number of Fourier modes
L = 30.0 # Space period
Δt = 2π / 256
t_end = 2π
Nsteps = round(Int, t_end / Δt)
config = NLSESettings(Nsteps, Δt, t_end, Ngrids, L)

t_ary = collect(0:config.Δt:config.t_end)[1:end - 1]
K = 4
naive_dp = solve_dmd(X, K)
X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)

X1, X2, X3, X4 = abs.(X), abs.(X_reconst_dmd), abs.(X_pointest_tbdmd), abs.(X_pointest_bdmdvmf)
cmin = minimum(hcat(X1, X2, X3, X4))
cmax = maximum(hcat(X1, X2, X3, X4))
p1 = heatmap(t_ary, config.gridpoints, X1,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p2 = heatmap(t_ary, config.gridpoints, X2,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p3 = heatmap(t_ary, config.gridpoints, X3,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p4 = heatmap(t_ary, config.gridpoints, X4,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
savefig(p1, "$outdir/NLSE_reconst_origin.pdf")
savefig(p2, "$outdir/NLSE_reconst_dmd.pdf")
savefig(p3, "$outdir/NLSE_reconst_tbdmd.pdf")
savefig(p4, "$outdir/NLSE_reconst_bdmd-vmf.pdf")

RMSEs[:NLSE][1] = √mean(abs2.(X - X_reconst_dmd))
RMSEs[:NLSE][2] = √mean(abs2.(X - X_pointest_tbdmd))
RMSEs[:NLSE][3] = √mean(abs2.(X - X_pointest_bdmdvmf))


### Burgers
@load "$outdir/mcmc_burgers_bdmdvmf.jld2" X X_pointest_bdmdvmf
@load "$outdir/mcmc_burgers_takeishi.jld2" X_pointest_tbdmd
Ngrids = 256 # Number of Fourier modes
L = 30.0 # Space period
t_end = 30.0
Δt = 1.0
Nsteps = round(Int, t_end / Δt)
ν = 0.1

config = BurgersSettings(Nsteps, Δt, t_end, Ngrids, L, ν)
t_ary = collect(0:config.Δt:config.t_end)

K = 7

naive_dp = solve_dmd(X, K)
X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)

X1, X2, X3, X4 = real.(X), real.(X_reconst_dmd), real.(X_pointest_tbdmd), real.(X_pointest_bdmdvmf)
cmin = minimum(hcat(X1, X2, X3, X4))
cmax = maximum(hcat(X1, X2, X3, X4))
p1 = heatmap(t_ary, config.gridpoints, X1,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p2 = heatmap(t_ary, config.gridpoints, X2,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p3 = heatmap(t_ary, config.gridpoints, X3,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
p4 = heatmap(t_ary, config.gridpoints, X4,
             colorbar = false, clims = (cmin, cmax), dpi = 200,
             xtickfontsize = 14, ytickfontsize = 14)
savefig(p1, "$outdir/burgers_reconst_origin.pdf")
savefig(p2, "$outdir/burgers_reconst_dmd.pdf")
savefig(p3, "$outdir/burgers_reconst_tbdmd.pdf")
savefig(p4, "$outdir/burgers_reconst_bdmd-vmf.pdf")

RMSEs[:Burgers][1] = √mean(abs2.(X - X_reconst_dmd))
RMSEs[:Burgers][2] = √mean(abs2.(X - X_pointest_tbdmd))
RMSEs[:Burgers][3] = √mean(abs2.(X - X_pointest_bdmdvmf))

CSV.write("$outdir/RMSEs.csv", RMSEs)
