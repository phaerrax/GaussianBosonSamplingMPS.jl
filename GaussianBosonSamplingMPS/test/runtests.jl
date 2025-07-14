using Test
using GaussianStates, ITensors, ITensorMPS, LinearAlgebra
using GaussianBosonSamplingMPS
using GaussianBosonSamplingMPS: hafnian, loophafnian, dirsum

# Check that the hafnian and loop-hafnian functions in the package have the expected scaling
# properties.
include("scaling_properties_hafnian.jl")

# Build some matrix-product states from Gaussian states of which we know (analytically) some
# features, e.g. the coefficients on the number basis, and check if what we get behaves
# accordingly.
include("correct_mps_construction.jl")

# Test the ITensor operators acting on the superbosonic matrix-product states that we
# defined, to see whether the functions run correctly and if the results make sense in some
# known cases.
include("operations_on_mps.jl")

# Transform (standard and superbosonic) matrix-product states into GaussianState objects and
# back, and see if everything checks out.
include("gaussian_states_mps_compatibility.jl")

# Test the various functions we defined for superbosonic matrix-product states to see if
# they run correctly and if the results make sense.
include("superbosons_mps.jl")
