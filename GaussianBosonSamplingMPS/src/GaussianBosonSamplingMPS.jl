module GaussianBosonSamplingMPS

using LinearAlgebra, GaussianStates, ITensors, ITensorMPS

# Sampling
using Distributions: MvNormal
using ProgressMeter

# Optimisation
using SCS, JuMP

include("normal_mode_decomposition.jl")
include("hafnian.jl")

export enlargelocaldim
# The `MPS` name is already exported by ITensorMPS
include("mps_construction.jl")

export SuperBosonMPS, measure, sb_siteinds, sb_index, inv_sb_index, sb_outer, sample
include("superbosons.jl")

export attenuate, firstmoments, covariancematrix, displace_pure
include("mps_operations.jl")

export optimise
include("state_optimisation.jl")

export sample_displaced
include("sampling.jl")

end # module GaussianBosonSamplingMPS
