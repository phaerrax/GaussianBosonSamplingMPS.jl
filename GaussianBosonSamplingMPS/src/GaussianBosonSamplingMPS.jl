module GaussianBosonSamplingMPS

using LinearAlgebra, GaussianStates, ITensors, ITensorMPS

include("normal_mode_decomposition.jl")
include("hafnian.jl")

# There's nothing we really need to export for now. The only function which is supposed
# to be accessed from the outside is `MPS`, whose name is already exported by ITensorMPS.
include("mps_construction.jl")

export displace!
include("sampling.jl")

export measure, sb_siteinds, sb_index, inv_sb_index
include("superbosons.jl")

export attenuate
include("mps_operations.jl")

end # module GaussianBosonSamplingMPS
