# BayesianDMDExamples
Illustrative Examples in the BDMD-VMF paper:  
Takahiro Kawashima, Hayaru Shouno, Hideitsu Hino, "Bayesian Dynamic Mode Decomposition with Variational Matrix Factorization," in AAAI-21.

# Working Directory
Please set your working directory `examples/`.

# Examples

|File|Description|Recommended environment(s)|
|:---|:----------|:-------------------------|
|`examples/examples_nlse.jl`|Example of the nonlinear Schörodinger equation|julia (≥ 1.5.0)|
|`examples/examples_burgers.jl`|Example of the Burgers' equation|julia (≥ 1.5.0)|
|`examples/hugadb.jl`|Example of gyroscope data (BDMD-VMF)|julia (≥ 1.5.0)|
|`examples/hugadb_var2.r`, `examples/hugadb_var2.stan`|Example of gyroscope data (VAR(2))|R (≥ 4.0.0)|


## Nonlinear Schörodinger Equation
Run `examples/examples_nlse.jl` to execute.
The data is simulated by the pseudo-spectral method.
Required libraries for Julia are below:

|Library|Recommended ver.|
|:------|:---------------|
|`Plots`  |≥ 1.5.8|
|`CSV`  |≥ 0.7.7|
|`DataFrames`  |≥ 0.21.6|
|`JLD2`  |≥ 0.1.14|
|`StatsBase`  |≥ 0.33.0|
|`PDMats` | ≥ 0.10.1|
|`ProgressMeter`  |≥ 1.3.2|
|`Distributions`  |≥ 0.23.8|
|`DifferentialEquations`  |≥ 6.15.0|
|`FFTW`  |≥ 1.2.2|
|`Random`  ||
|`LinearAlgebra`  ||

## Burgers' Equation
Run `examples/examples_burgers.jl` to execute.
The data is simulated by the Crank-Nicolson method.
Required libraries for Julia are below:

|Library|Recommended ver.|
|:------|:---------------|
|`Plots`  |≥ 1.5.8|
|`CSV`  |≥ 0.7.7|
|`DataFrames`  |≥ 0.21.6|
|`JLD2`  |≥ 0.1.14|
|`StatsBase`  |≥ 0.33.0|
|`PDMats` | ≥ 0.10.1|
|`ProgressMeter`  |≥ 1.3.2|
|`Distributions`  |≥ 0.23.8|
|`Random`  ||
|`LinearAlgebra`  ||



## Gyroscope Data
Run `examples/hugadb.jl` to execute.
To obtain a result by the Bayesian VAR(2) model,
execute `examples/hugadb_var2.r` before running `examples/hugadb.jl`.
Required libraries for Julia are below:

|Library|Recommended ver.|
|:------|:---------------|
|`Plots`  |≥ 1.5.8|
|`CSV`  |≥ 0.7.7|
|`DataFrames`  |≥ 0.21.6|
|`JLD2`  |≥ 0.1.14|
|`StatsBase`  |≥ 0.33.0|
|`PDMats` | ≥ 0.10.1|
|`ProgressMeter`  |≥ 1.3.2|
|`Distributions`  |≥ 0.23.8|
|`KernelDensity` |≥ 0.6.0|
|`Random`  ||
|`LinearAlgebra`  ||

In addition, required library for R are below:

|Library|Recommended ver.|
|:------|:---------------|
|`rstan`  |≥ 2.21.2|
|`dplyr`  |≥ 1.0.0|
