using Documenter
using GaussianBosonSamplingMPS

push!(LOAD_PATH,"../src/")
# This line is needed because the GaussianBosonSamplingMPS is not accessible through Julia's
# LOAD_PATH.

makedocs(
    sitename = "GaussianBosonSamplingMPS",
    format = Documenter.HTML(),
    modules = [GaussianBosonSamplingMPS],
    remotes = nothing
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
